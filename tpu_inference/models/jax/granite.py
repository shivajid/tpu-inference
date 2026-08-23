# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""IBM Granite (GraniteForCausalLM) JAX model for TPU inference.

Granite is architecturally Llama-like (GQA, RMSNorm, SwiGLU, RoPE) but adds
four scaling multipliers that must be applied exactly as in the HF reference
(transformers.models.granite.modeling_granite), otherwise outputs are wrong:

  * attention_multiplier  -> replaces the default 1/sqrt(head_dim) softmax scale
  * embedding_multiplier  -> scales token embeddings at the model input
  * residual_multiplier   -> scales each residual branch (attn and MLP) output
  * logits_scaling        -> final logits are divided by this before softmax

Weight layout and HF<->nnx mappings are identical to Llama, so we reuse the
StandardWeightLoader with the same name mapping.
"""

from itertools import islice
from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh
from transformers import GraniteConfig
from vllm.config import VllmConfig

from tpu_inference import utils
from tpu_inference.distributed.jax_parallel_state import get_pp_group
from tpu_inference.layers.common.attention_interface import attention
from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.layers.common.quantization import quantize_kv
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.layers.jax.layers import FlaxUtils
from tpu_inference.layers.jax.pp_utils import PPMissingLayer, make_layers
from tpu_inference.layers.jax.rope_interface import (apply_rope,
                                                     get_rope_scaling,
                                                     get_rope_theta)
from tpu_inference.logger import init_logger
from tpu_inference.models.jax.jax_intermediate_tensor import \
    JaxIntermediateTensors
from tpu_inference.models.jax.utils.weight_utils import StandardWeightLoader
from tpu_inference.utils import get_mesh_shape_product

logger = init_logger(__name__)

init_fn = nnx.initializers.uniform()
modeling_flax_utils = FlaxUtils()


class GraniteMLP(nnx.Module):

    def __init__(self, config: GraniteConfig, dtype: jnp.dtype,
                 rng: nnx.Rngs):
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size
        act = config.hidden_act

        self.gate_proj = nnx.Linear(
            hidden_size,
            intermediate_size,
            use_bias=False,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(
                init_fn, (None, ShardingAxisName.MLP_TENSOR)),
            rngs=rng,
        )
        self.up_proj = nnx.Linear(
            hidden_size,
            intermediate_size,
            use_bias=False,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(
                init_fn, (None, ShardingAxisName.MLP_TENSOR)),
            rngs=rng,
        )
        self.down_proj = nnx.Linear(
            intermediate_size,
            hidden_size,
            use_bias=False,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(
                init_fn, (ShardingAxisName.MLP_TENSOR, None)),
            rngs=rng,
        )
        self.act_fn = modeling_flax_utils.ACT2FN[act]

    def __call__(self, x: jax.Array) -> jax.Array:
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        fuse = gate * up
        result = self.down_proj(fuse)
        return result


class GraniteAttention(nnx.Module):

    def __init__(self, config: GraniteConfig, dtype: jnp.dtype,
                 rng: nnx.Rngs, mesh: Mesh, kv_cache_dtype: str):
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.rope_theta = get_rope_theta(config, default=10000000.0)
        self.rope_scaling = get_rope_scaling(config)

        self.head_dim_original = getattr(config, "head_dim",
                                         self.hidden_size // self.num_heads)
        self.head_dim = utils.get_padded_head_dim(self.head_dim_original)

        sharding_size = get_mesh_shape_product(mesh,
                                               ShardingAxisName.MLP_TENSOR)
        self.num_heads = utils.get_padded_num_heads(self.num_heads,
                                                    sharding_size)
        self.num_kv_heads = utils.get_padded_num_heads(self.num_kv_heads,
                                                       sharding_size)

        self.mesh = mesh

        # Granite uses its own attention multiplier in place of 1/sqrt(head_dim).
        self.attention_multiplier = config.attention_multiplier

        self.q_proj = nnx.Einsum(
            "TD,DNH->TNH",
            (self.hidden_size, self.num_heads, self.head_dim),
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(
                init_fn, (None, ShardingAxisName.ATTN_HEAD, None)),
            rngs=rng,
        )
        self.k_proj = nnx.Einsum(
            "TD,DKH->TKH",
            (self.hidden_size, self.num_kv_heads, self.head_dim),
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(
                init_fn, (None, ShardingAxisName.ATTN_HEAD, None)),
            rngs=rng,
        )
        self.v_proj = nnx.Einsum(
            "TD,DKH->TKH",
            (self.hidden_size, self.num_kv_heads, self.head_dim),
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(
                init_fn, (None, ShardingAxisName.ATTN_HEAD, None)),
            rngs=rng,
        )
        self.o_proj = nnx.Einsum(
            "TNH,NHD->TD",
            (self.num_heads, self.head_dim, self.hidden_size),
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(
                init_fn, (ShardingAxisName.ATTN_HEAD, None, None)),
            rngs=rng,
        )

        self._q_scale = 1.0
        self._k_scale = 1.0
        self._v_scale = 1.0
        self.kv_cache_quantized_dtype = None
        if kv_cache_dtype != "auto":
            self.kv_cache_quantized_dtype = utils.get_jax_dtype_from_str_dtype(
                kv_cache_dtype)

    def __call__(
        self,
        kv_cache: Optional[jax.Array],
        x: jax.Array,
        attention_metadata: AttentionMetadata,
    ) -> Tuple[jax.Array, jax.Array]:
        md = attention_metadata
        # q: (T, N, H)
        q = self.q_proj(x)
        q = apply_rope(q, md.input_positions, self.head_dim_original,
                       self.rope_theta, self.rope_scaling)
        # k: (T, K, H)
        k = self.k_proj(x)
        k = apply_rope(k, md.input_positions, self.head_dim_original,
                       self.rope_theta, self.rope_scaling)
        # v: (T, K, H)
        v = self.v_proj(x)
        # o: (T, N, H)
        q_scale = k_scale = v_scale = None
        if self.kv_cache_quantized_dtype:
            k_scale = self._k_scale
            v_scale = self._v_scale
            k, v = quantize_kv(self.kv_cache_quantized_dtype, k, v, k_scale,
                               v_scale)
        new_kv_cache, outputs = attention(
            kv_cache,
            q,
            k,
            v,
            attention_metadata,
            self.mesh,
            self.head_dim_original,
            sm_scale=self.attention_multiplier,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
        )
        # (T, D)
        o = self.o_proj(outputs)
        return new_kv_cache, o


class GraniteDecoderLayer(nnx.Module):

    def __init__(self, config: GraniteConfig, dtype: jnp.dtype,
                 rng: nnx.Rngs, mesh: Mesh, kv_cache_dtype: str):
        rms_norm_eps = config.rms_norm_eps
        hidden_size = config.hidden_size
        self.residual_multiplier = config.residual_multiplier

        self.input_layernorm = nnx.RMSNorm(
            hidden_size,
            epsilon=rms_norm_eps,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None, )),
            rngs=rng,
        )
        self.self_attn = GraniteAttention(config=config,
                                          dtype=dtype,
                                          rng=rng,
                                          mesh=mesh,
                                          kv_cache_dtype=kv_cache_dtype)
        self.post_attention_layernorm = nnx.RMSNorm(
            hidden_size,
            epsilon=rms_norm_eps,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None, )),
            rngs=rng,
        )
        self.mlp = GraniteMLP(
            config=config,
            dtype=dtype,
            rng=rng,
        )

    def __call__(
        self,
        kv_cache: jax.Array,
        x: jax.Array,
        attention_metadata: AttentionMetadata,
    ) -> Tuple[jax.Array, jax.Array]:
        # residual + attn_out * residual_multiplier  (matches HF reference)
        hidden_states = self.input_layernorm(x)
        kv_cache, attn_output = self.self_attn(
            kv_cache,
            hidden_states,
            attention_metadata,
        )
        attn_output = x + attn_output * self.residual_multiplier

        # residual + mlp_out * residual_multiplier
        residual = attn_output
        hidden_states = self.post_attention_layernorm(attn_output)
        outputs = self.mlp(hidden_states)
        outputs = residual + outputs * self.residual_multiplier
        return kv_cache, outputs


class GraniteModel(nnx.Module):

    def __init__(self, vllm_config: VllmConfig, rng: nnx.Rngs,
                 mesh: Mesh) -> None:
        model_config = vllm_config.model_config
        hf_config = model_config.hf_config
        self.vocab_size_original = model_config.get_vocab_size()
        self.vocab_size_padded = utils.get_padded_vocab_size(
            self.vocab_size_original)
        vocab_size = self.vocab_size_padded
        dtype = model_config.dtype
        rms_norm_eps = hf_config.rms_norm_eps
        hidden_size = hf_config.hidden_size

        self.embedding_multiplier = hf_config.embedding_multiplier
        self.is_first_rank = get_pp_group().is_first_rank
        self.is_last_rank = get_pp_group().is_last_rank

        if self.is_first_rank or (hf_config.tie_word_embeddings
                                  and self.is_last_rank):
            self.embed = nnx.Embed(
                num_embeddings=vocab_size,
                features=hidden_size,
                param_dtype=dtype,
                embedding_init=nnx.with_partitioning(
                    init_fn, (ShardingAxisName.VOCAB, None)),
                rngs=rng,
            )
        else:
            self.embed = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            hf_config.num_hidden_layers,
            lambda _: GraniteDecoderLayer(
                config=hf_config,
                dtype=dtype,
                rng=rng,
                mesh=mesh,
                kv_cache_dtype=vllm_config.cache_config.cache_dtype))
        if self.is_last_rank:
            self.norm = nnx.RMSNorm(
                hidden_size,
                epsilon=rms_norm_eps,
                param_dtype=dtype,
                scale_init=nnx.with_partitioning(init_fn, (None, )),
                rngs=rng,
            )
        else:
            self.norm = PPMissingLayer()

        if self.is_last_rank:
            if model_config.hf_config.tie_word_embeddings:
                self.lm_head = self.embed.embedding
            else:
                self.lm_head = nnx.Param(
                    init_fn(rng.params(), (hidden_size, vocab_size), dtype),
                    sharding=(None, ShardingAxisName.VOCAB),
                )
        else:
            self.lm_head = PPMissingLayer()

    def __call__(
        self,
        kv_caches: List[jax.Array],
        input_ids: jax.Array,
        attention_metadata: AttentionMetadata,
        intermediate_tensors: JaxIntermediateTensors | None,
    ) -> Tuple[List[jax.Array], jax.Array | JaxIntermediateTensors,
               List[jax.Array]]:
        if self.is_first_rank:
            # scale embeddings at the model input (matches HF reference)
            x = self.embed(input_ids) * self.embedding_multiplier
        else:
            assert intermediate_tensors is not None
            x = intermediate_tensors["hidden_states"]

        aux_hidden_states: List[jax.Array] = []
        for i, layer in enumerate(
                islice(self.layers, self.start_layer, self.end_layer)):
            kv_cache = kv_caches[i]
            kv_cache, x = layer(
                kv_cache,
                x,
                attention_metadata,
            )
            kv_caches[i] = kv_cache
        if not self.is_last_rank:
            return kv_caches, JaxIntermediateTensors({"hidden_states":
                                                      x}), aux_hidden_states
        x = self.norm(x)
        return kv_caches, x, aux_hidden_states


class GraniteWeightLoader(StandardWeightLoader):
    """StandardWeightLoader that also zero-pads the embedding vocab axis.

    Granite's vocab (49159) is not divisible by common TP degrees, and the
    tied embedding/lm_head is sharded along the vocab axis. We inject a vocab
    zero-pad into the metadata map's pad_map so the shared loader pads the
    embedding weight up to the padded (128-multiple) vocab size before it is
    sharded. Padded rows correspond to token ids that are never produced by
    the tokenizer, so they are unreachable at inference time.
    """

    def __init__(self, vllm_config: VllmConfig, mesh: Mesh,
                 vocab_size_original: int, vocab_size_padded: int):
        super().__init__(vllm_config, mesh)
        self.vocab_size_original = vocab_size_original
        self.vocab_size_padded = vocab_size_padded

    def load_weights(self, model: nnx.Module, mappings: dict,
                     keep_hf_weight_suffix_when_match: list[str] = []):
        from tpu_inference.models.jax.utils.weight_utils import get_default_maps
        metadata_map = get_default_maps(self.vllm_config.model_config,
                                        self.mesh, mappings)
        vocab_pad = self.vocab_size_padded - self.vocab_size_original
        if vocab_pad > 0:
            # (pad_dim, pad_size); axis 0 is the vocab axis for both
            # embed_tokens (vocab, hidden) and lm_head (post-transpose).
            metadata_map.vocab_pad_map["embed_tokens"] = (0, vocab_pad)
            metadata_map.vocab_pad_map["lm_head"] = (0, vocab_pad)
        super().load_weights(
            model, metadata_map,
            keep_hf_weight_suffix_when_match=keep_hf_weight_suffix_when_match)


class GraniteForCausalLM(nnx.Module):
    WeightLoader = GraniteWeightLoader

    def __init__(self, vllm_config: VllmConfig, rng_key: jax.Array,
                 mesh: Mesh) -> None:
        self.vllm_config = vllm_config
        self.rng = nnx.Rngs(rng_key)
        self.mesh = mesh

        self.model = GraniteModel(
            vllm_config=vllm_config,
            rng=self.rng,
            mesh=mesh,
        )

        self.pp_missing_layers = []
        for path, module in nnx.iter_graph(self.model):
            if isinstance(module, PPMissingLayer):
                self.pp_missing_layers.append('.'.join([str(s) for s in path]))

    def __call__(
        self,
        kv_caches: List[jax.Array],
        input_ids: jax.Array,
        attention_metadata: AttentionMetadata,
        _input_embeds=None,
        _input_positions=None,
        _layer_name_to_kv_cache=None,
        _lora_metadata=None,
        intermediate_tensors: JaxIntermediateTensors | None = None,
        _is_first_rank: bool | None = None,
        _is_last_rank: bool | None = None,
        *args,
    ) -> Tuple[List[jax.Array], jax.Array | JaxIntermediateTensors,
               List[jax.Array], Optional[jax.Array]]:
        outputs = self.model(
            kv_caches,
            input_ids,
            attention_metadata,
            intermediate_tensors,
        )
        return outputs[0], outputs[1], outputs[2], None

    def compute_logits(self, hidden_states: jax.Array) -> jax.Array:
        if self.vllm_config.model_config.hf_config.tie_word_embeddings:
            logits = jnp.dot(hidden_states, self.model.lm_head.value.T)
        else:
            logits = jnp.dot(hidden_states, self.model.lm_head.value)
        # divide by logits_scaling before softmax (matches HF reference)
        logits = logits / self.vllm_config.model_config.hf_config.logits_scaling
        return logits

    def load_weights(self, rng_key: jax.Array):
        # NOTE: Since we are using nnx.eval_shape to init the model,
        # we have to pass dynamic arrays here for __call__'s usage.
        self.rng = nnx.Rngs(rng_key)

        # Key: path to a HF layer weight
        # Value: path to a nnx layer weight
        mappings = {
            "model.embed_tokens": "model.embed.embedding",
            "model.layers.*.input_layernorm":
            "model.layers.*.input_layernorm.scale",
            "model.layers.*.mlp.down_proj":
            "model.layers.*.mlp.down_proj.kernel",
            "model.layers.*.mlp.gate_proj":
            "model.layers.*.mlp.gate_proj.kernel",
            "model.layers.*.mlp.up_proj": "model.layers.*.mlp.up_proj.kernel",
            "model.layers.*.post_attention_layernorm":
            "model.layers.*.post_attention_layernorm.scale",
            "model.layers.*.self_attn.k_proj":
            "model.layers.*.self_attn.k_proj.kernel",
            "model.layers.*.self_attn.o_proj":
            "model.layers.*.self_attn.o_proj.kernel",
            "model.layers.*.self_attn.q_proj":
            "model.layers.*.self_attn.q_proj.kernel",
            "model.layers.*.self_attn.v_proj":
            "model.layers.*.self_attn.v_proj.kernel",
            "model.norm": "model.norm.scale",
        }
        # Add lm_head mapping only if it's not tied to embeddings
        if not self.vllm_config.model_config.hf_config.tie_word_embeddings:
            mappings.update({
                "lm_head": "model.lm_head",
            })

        loader = self.WeightLoader(
            self.vllm_config,
            self.mesh,
            vocab_size_original=self.model.vocab_size_original,
            vocab_size_padded=self.model.vocab_size_padded,
        )
        loader.load_weights(self, mappings)
