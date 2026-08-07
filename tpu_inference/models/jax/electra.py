# Copyright 2026 Google LLC
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
"""JAX-native ELECTRA encoder-only embedding model.

Implements the ELECTRA discriminator encoder (`google/electra-small-*`,
arch string "ElectraForPreTraining"): a BERT-style post-LayerNorm encoder
with learned absolute position embeddings and a factorized embedding table
(`embedding_size` != `hidden_size`, bridged by `embeddings_project`).

The replaced-token-detection head (`discriminator_predictions.*`) is not
instantiated — this integration serves the pretrained encoder as an
embedding model. Runs prefill-only under `--runner pooling`; the pooler
executes on CPU via vLLM's DispatchPooler (see model_loader). No KV cache.

Module attribute names deliberately mirror the HF checkpoint tensor names
(after stripping the "electra." prefix): `embeddings.word_embeddings.weight`,
`embeddings_project.weight`, `encoder.layer.N.attention.self.query.weight`,
`encoder.layer.N.intermediate.dense.weight`, etc., so that
`JaxAutoWeightsLoader` matches them without a rename table.
"""

import functools
from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh
from vllm.config import VllmConfig

from tpu_inference import utils
from tpu_inference.layers.common.attention_interface import \
    encoder_only_attention
from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.layers.jax import JaxModule, JaxModuleList
from tpu_inference.layers.jax.embed import JaxEmbed
from tpu_inference.layers.jax.linear import JaxEinsum, JaxLinear
from tpu_inference.layers.jax.norm import JaxLayerNorm
from tpu_inference.logger import init_logger
from tpu_inference.models.jax.jax_intermediate_tensor import \
    JaxIntermediateTensors
from tpu_inference.models.jax.utils.weight_utils import (
    LoadableWithIterator, load_nnx_param_from_reshaped_torch)

logger = init_logger(__name__)

init_fn = nnx.initializers.uniform()


def _set_weight_loader(param: nnx.Param,
                       param_name: str,
                       reshape_dims: Optional[Tuple[int, ...]] = None,
                       permute_dims: Optional[Tuple[int, ...]] = None) -> None:
    """Attach an explicit HF->JAX weight loader to a param.

    Needed where the auto-loader's name-based heuristics (q_proj/o_proj/
    embed_tokens) don't match the HF BERT-style names used here.
    """
    param.set_metadata(
        "weight_loader",
        functools.partial(load_nnx_param_from_reshaped_torch,
                          reshape_dims=reshape_dims,
                          permute_dims=permute_dims,
                          param_name=param_name))


class ElectraEmbeddings(JaxModule):
    """word + position + token_type embeddings -> LayerNorm.

    All tables live in `embedding_size` (128 for electra-small), not
    `hidden_size`; the projection to `hidden_size` happens afterwards in
    `ElectraModel.embeddings_project`.
    """

    def __init__(self, config, dtype: jnp.dtype, rng: nnx.Rngs):
        embedding_size = getattr(config, "embedding_size", config.hidden_size)
        self.word_embeddings = JaxEmbed(
            num_embeddings=config.vocab_size,
            features=embedding_size,
            dtype=dtype,
            embedding_init=nnx.with_partitioning(init_fn, ("model", None)),
            rngs=rng,
        )
        self.position_embeddings = JaxEmbed(
            num_embeddings=config.max_position_embeddings,
            features=embedding_size,
            dtype=dtype,
            embedding_init=nnx.with_partitioning(init_fn, (None, None)),
            rngs=rng,
        )
        self.token_type_embeddings = JaxEmbed(
            num_embeddings=config.type_vocab_size,
            features=embedding_size,
            dtype=dtype,
            embedding_init=nnx.with_partitioning(init_fn, (None, None)),
            rngs=rng,
        )
        self.LayerNorm = JaxLayerNorm(
            num_features=embedding_size,
            epsilon=config.layer_norm_eps,
            dtype=dtype,
            rngs=rng,
        )
        # Embedding tables must be loaded as-is (no 2D transpose).
        _set_weight_loader(self.word_embeddings.weight,
                           "embeddings.word_embeddings.weight",
                           permute_dims=(0, 1))
        _set_weight_loader(self.position_embeddings.weight,
                           "embeddings.position_embeddings.weight",
                           permute_dims=(0, 1))
        _set_weight_loader(self.token_type_embeddings.weight,
                           "embeddings.token_type_embeddings.weight",
                           permute_dims=(0, 1))

    def __call__(self, input_ids: jax.Array,
                 positions: jax.Array) -> jax.Array:
        x = self.word_embeddings(input_ids)
        x = x + self.position_embeddings(positions)
        # Embedding use: token_type_ids are all zeros -> row 0 broadcast.
        x = x + self.token_type_embeddings.weight.value[0]
        return self.LayerNorm(x)


class ElectraSelfAttention(JaxModule):
    """QKV projections + encoder-only flash attention (no positional bias:
    ELECTRA uses learned absolute position embeddings, applied upstream)."""

    def __init__(self, config, dtype: jnp.dtype, rng: nnx.Rngs, mesh: Mesh,
                 prefix: str):
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim_original = self.hidden_size // self.num_heads
        self.head_dim = utils.get_padded_head_dim(self.head_dim_original)
        sharding_size = mesh.shape["model"]
        self.num_heads = utils.get_padded_num_heads(self.num_heads,
                                                    sharding_size)
        self.mesh = mesh

        def qkv(name):
            proj = JaxEinsum(
                "TD,DNH->TNH",
                (self.hidden_size, self.num_heads, self.head_dim),
                bias_shape=(self.num_heads, self.head_dim),
                dtype=dtype,
                kernel_init=nnx.with_partitioning(init_fn,
                                                  (None, "model", None)),
                bias_init=nnx.with_partitioning(init_fn, ("model", None)),
                rngs=rng,
                prefix=f"{prefix}.{name}",
            )
            # HF: [N*H, D] -> reshape (N, H, D) -> permute to (D, N, H).
            _set_weight_loader(proj.weight,
                               f"{prefix}.{name}.weight",
                               reshape_dims=(self.num_heads, self.head_dim,
                                             self.hidden_size),
                               permute_dims=(2, 0, 1))
            _set_weight_loader(proj.bias,
                               f"{prefix}.{name}.bias",
                               reshape_dims=(self.num_heads, self.head_dim),
                               permute_dims=(0, 1))
            return proj

        self.query = qkv("query")
        self.key = qkv("key")
        self.value = qkv("value")

    def __call__(self, x: jax.Array,
                 attention_metadata: AttentionMetadata) -> jax.Array:
        q = self.query(x)  # [T, N, H]
        k = self.key(x)
        v = self.value(x)
        return encoder_only_attention(
            q,
            k,
            v,
            attention_metadata,
            self.mesh,
            sm_scale=self.head_dim_original**-0.5,
        )


class ElectraSelfOutput(JaxModule):
    """attention output dense + residual + post-LayerNorm."""

    def __init__(self, config, dtype: jnp.dtype, rng: nnx.Rngs,
                 num_heads: int, head_dim: int, prefix: str):
        hidden_size = config.hidden_size
        self.dense = JaxEinsum(
            "TNH,NHD->TD",
            (num_heads, head_dim, hidden_size),
            bias_shape=(hidden_size, ),
            dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, ("model", None, None)),
            bias_init=nnx.with_partitioning(init_fn, (None, )),
            rngs=rng,
            prefix=prefix + ".dense",
        )
        # HF: [D_out, N*H] -> reshape (D, N, H) -> permute to (N, H, D).
        _set_weight_loader(self.dense.weight,
                           f"{prefix}.dense.weight",
                           reshape_dims=(hidden_size, num_heads, head_dim),
                           permute_dims=(1, 2, 0))
        self.LayerNorm = JaxLayerNorm(
            num_features=hidden_size,
            epsilon=config.layer_norm_eps,
            dtype=dtype,
            rngs=rng,
        )

    def __call__(self, x: jax.Array, residual: jax.Array) -> jax.Array:
        return self.LayerNorm(self.dense(x) + residual)


class ElectraAttention(JaxModule):

    def __init__(self, config, dtype: jnp.dtype, rng: nnx.Rngs, mesh: Mesh,
                 prefix: str):
        self_attention = ElectraSelfAttention(config,
                                              dtype,
                                              rng,
                                              mesh,
                                              prefix=prefix + ".self")
        # HF checkpoint path is `attention.self.*`; `self` is a valid
        # attribute name in Python.
        setattr(self, "self", self_attention)
        self.output = ElectraSelfOutput(
            config,
            dtype,
            rng,
            num_heads=self_attention.num_heads,
            head_dim=self_attention.head_dim,
            prefix=prefix + ".output",
        )

    def __call__(self, x: jax.Array,
                 attention_metadata: AttentionMetadata) -> jax.Array:
        attn = getattr(self, "self")(x, attention_metadata)
        return self.output(attn, x)


class ElectraIntermediate(JaxModule):
    """hidden -> intermediate dense + GELU (standard BERT MLP up-proj)."""

    def __init__(self, config, dtype: jnp.dtype, rng: nnx.Rngs, prefix: str):
        assert config.hidden_act in ("gelu", "gelu_new"), (
            f"Unsupported hidden_act {config.hidden_act!r}")
        # transformers "gelu" is the exact erf form; "gelu_new" is tanh.
        self._approximate = config.hidden_act == "gelu_new"
        self.dense = JaxLinear(
            config.hidden_size,
            config.intermediate_size,
            use_bias=True,
            dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model")),
            bias_init=nnx.with_partitioning(init_fn, ("model", )),
            rngs=rng,
            prefix=prefix + ".dense",
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        return jax.nn.gelu(self.dense(x), approximate=self._approximate)


class ElectraOutput(JaxModule):
    """intermediate -> hidden dense + residual + post-LayerNorm."""

    def __init__(self, config, dtype: jnp.dtype, rng: nnx.Rngs, prefix: str):
        self.dense = JaxLinear(
            config.intermediate_size,
            config.hidden_size,
            use_bias=True,
            dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, ("model", None)),
            bias_init=nnx.with_partitioning(init_fn, (None, )),
            rngs=rng,
            prefix=prefix + ".dense",
        )
        self.LayerNorm = JaxLayerNorm(
            num_features=config.hidden_size,
            epsilon=config.layer_norm_eps,
            dtype=dtype,
            rngs=rng,
        )

    def __call__(self, x: jax.Array, residual: jax.Array) -> jax.Array:
        return self.LayerNorm(self.dense(x) + residual)


class ElectraLayer(JaxModule):

    def __init__(self, config, dtype: jnp.dtype, rng: nnx.Rngs, mesh: Mesh,
                 prefix: str):
        self.attention = ElectraAttention(config,
                                          dtype,
                                          rng,
                                          mesh,
                                          prefix=prefix + ".attention")
        self.intermediate = ElectraIntermediate(config,
                                                dtype,
                                                rng,
                                                prefix=prefix +
                                                ".intermediate")
        self.output = ElectraOutput(config,
                                    dtype,
                                    rng,
                                    prefix=prefix + ".output")

    def __call__(self, x: jax.Array,
                 attention_metadata: AttentionMetadata) -> jax.Array:
        x = self.attention(x, attention_metadata)
        return self.output(self.intermediate(x), x)


class ElectraEncoder(JaxModule):

    def __init__(self, config, dtype: jnp.dtype, rng: nnx.Rngs, mesh: Mesh,
                 prefix: str):
        # nnx.List container: Flax >= 0.12 rejects plain Python lists of
        # modules as pytree attributes.
        self.layer = JaxModuleList([
            ElectraLayer(config, dtype, rng, mesh, prefix=f"{prefix}.layer.{i}")
            for i in range(config.num_hidden_layers)
        ])

    def __call__(self, x: jax.Array,
                 attention_metadata: AttentionMetadata) -> jax.Array:
        for layer in self.layer:
            x = layer(x, attention_metadata)
        return x


class ElectraModel(JaxModule):

    def __init__(self, vllm_config: VllmConfig, rng: nnx.Rngs, mesh: Mesh):
        config = vllm_config.model_config.hf_config
        dtype = vllm_config.model_config.dtype
        embedding_size = getattr(config, "embedding_size", config.hidden_size)
        self.embeddings = ElectraEmbeddings(config, dtype, rng)
        if embedding_size != config.hidden_size:
            self.embeddings_project = JaxLinear(
                embedding_size,
                config.hidden_size,
                use_bias=True,
                dtype=dtype,
                kernel_init=nnx.with_partitioning(init_fn, (None, None)),
                bias_init=nnx.with_partitioning(init_fn, (None, )),
                rngs=rng,
                prefix="embeddings_project",
            )
        else:
            self.embeddings_project = None
        self.encoder = ElectraEncoder(config, dtype, rng, mesh,
                                      prefix="encoder")

    def __call__(self, input_ids: jax.Array, positions: jax.Array,
                 attention_metadata: AttentionMetadata) -> jax.Array:
        x = self.embeddings(input_ids, positions)
        if self.embeddings_project is not None:
            x = self.embeddings_project(x)
        return self.encoder(x, attention_metadata)


class ElectraForPreTraining(JaxModule, LoadableWithIterator):
    """Embedding-only ELECTRA discriminator ("ElectraForPreTraining" arch).

    The replaced-token-detection head (`discriminator_predictions.*`) is not
    instantiated; pooling runs in vLLM's CPU pooler.
    """

    # vLLM registry inspection: this model only supports the pooling runner.
    is_pooling_model = True

    def __init__(self, vllm_config: VllmConfig, rng_key: jax.Array,
                 mesh: Mesh) -> None:
        self.vllm_config = vllm_config
        self.mesh = mesh
        rng = nnx.Rngs(rng_key)
        self.model = ElectraModel(vllm_config, rng, mesh)

    def __call__(
        self,
        kv_caches: List[jax.Array],
        input_ids: jax.Array,
        attention_metadata: AttentionMetadata,
        inputs_embeds: Optional[jax.Array] = None,
        _input_positions=None,
        _layer_name_to_kv_cache=None,
        _lora_metadata=None,
        intermediate_tensors: Optional[JaxIntermediateTensors] = None,
        is_first_rank: bool = True,
        is_last_rank: bool = True,
        *args,
    ) -> Tuple[List[jax.Array], jax.Array, List[jax.Array],
               Optional[jax.Array]]:
        assert inputs_embeds is None, (
            "Electra does not support external input embeddings")
        # Learned absolute position embeddings need per-token positions;
        # the runner provides them in the attention metadata (0-based per
        # sequence in the concatenated ragged batch).
        positions = attention_metadata.input_positions
        hidden_states = self.model(input_ids, positions, attention_metadata)
        # Encoder-only: kv_caches pass through untouched.
        return kv_caches, hidden_states, [], None

    def load_weights(self, weights):
        """Strip the 'electra.' prefix, drop the RTD head (and any generator
        weights), then delegate to the standard JAX auto-loader."""
        from tpu_inference.models.jax.utils.weight_utils import \
            JaxAutoWeightsLoader
        from tpu_inference.utils import to_torch_dtype

        # Cast checkpoint dtype to the requested --dtype on load.
        torch_dtype = to_torch_dtype(self.vllm_config.model_config.dtype)

        def _filtered(weights_iter):
            for name, weight in weights_iter:
                if name.startswith("electra."):
                    name = name[len("electra."):]
                if name.startswith(("discriminator_predictions.", "generator",
                                    "cls.", "pooler.")):
                    continue
                yield name, weight.to(torch_dtype)

        loader = JaxAutoWeightsLoader(self, skip_prefixes=None)
        return loader.load_weights(_filtered(weights))
