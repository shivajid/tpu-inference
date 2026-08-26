# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TPU (torchax) entrypoint for GLM-5.3-Flash (``glm5_next``).

GLM-5.3-Flash (zai-org/GLM-5.3-Flash) is a hybrid-architecture MoE model:

* 45 decoder layers alternating KDA (Kimi Delta Attention, a gated linear
  attention with short conv1d, ``layer_types[i] == "linear_attention"``) and
  DeepSeek-style MLA layers with a DSA (sparse) indexer
  (``layer_types[i] == "deepseek_sparse_attention"``).
* Manifold-constrained Hyper-Connections (mHC, hc_mult=4) around both the
  attention and FFN sublayers, identical in structure to DeepSeek-V4.
* A 288-expert sparse MoE (sigmoid router, ``noaux_tc`` top-k) with one shared
  expert for most layers; the first ``first_k_dense_replace`` layers use a
  dense MLP.
* Weights are fp8 (block-wise e4m3); attention / router / mHC / norms stay in
  bf16/fp32 per the checkpoint's ``modules_to_not_convert``.

This module is a native TPU bring-up that composes existing torchax-compatible
pieces:

* KDA layers reuse vLLM's ``KimiGatedDeltaNetAttention`` (the Kimi-Linear KDA
  block), which the TPU platform routes to the registered ``gdn_attention``
  custom-op implementation
  (``tpu_inference.layers.vllm.custom_ops.gdn_attention_op``).
* MLA layers reuse ``DeepseekV2MLAAttention`` whose ``mla_attn`` is swapped to
  the TPU Pallas MLA backend by the register_oot MLA wrapper. The DSA
  lightning indexer is bypassed for the initial dense bring-up
  (``GLM_DSA_DISABLE=1``, default on in this wrapper).
* MoE / MLP / mHC reuse the DeepSeek-V4 TPU port
  (``tpu_inference.models.vllm.experimental.deepseek_v4``), which already
  implements the exact mHC parameterization (hc_attn_fn/base/scale,
  hc_ffn_fn/base/scale) and the sigmoid/noaux_tc FusedMoE wiring.

Only the text backbone (``Glm5NextTextForCausalLM``) is brought up here; the
vision tower of ``Glm5NextForConditionalGeneration`` is a follow-up
milestone.
"""

import typing
from collections.abc import Iterable

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mhc import (HCHeadOp, MHCFusedPostPreOp,
                                            MHCPostOp, MHCPreOp)
from tpu_inference.layers.vllm.custom_ops.kda_attention_op import (
    VllmKimiGatedDeltaNetAttention)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.deepseek_v2 import (
    DeepseekV2MLAAttention as _UpstreamDeepseekV2MLAAttention)
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.utils import (
    AutoWeightsLoader, PPMissingLayer, extract_layer_index,
    is_pp_missing_parameter, make_layers, maybe_prefix)
from vllm.sequence import IntermediateTensors

from tpu_inference import envs
from tpu_inference.logger import init_logger
from tpu_inference.models.vllm.experimental.deepseek_v4 import (
    DeepseekV4MLP, DeepseekV4MoE)

logger = init_logger(__name__)


def _text_config(config):
    """Unwrap the multimodal Glm5NextConfig to its text_config sub-config.

    zai-org/GLM-5.3-Flash ships a multimodal config (``Glm5NextConfig``) whose
    language-model params live under ``.text_config``; the text-only
    ``Glm5NextTextConfig`` has them at top level. All blocks in this port need
    the text config.
    """
    config = getattr(config, "text_config", config)
    # GLM-5.3 uses pure-nope MLA (qk_rope_head_dim=0) with no rope config, but
    # vLLM's DeepseekV2MLAAttention unconditionally reads
    # config.rope_parameters["rope_type"]. Provide a default.
    if getattr(config, "rope_parameters", None) is None:
        config.rope_parameters = {"rope_type": "default"}
    # DeepseekV4MoE reads config.num_hash_layers (a DeepSeek-V4 hash-MoE
    # field); GLM-5.3 has no hash-MoE layers, so default to 0.
    if getattr(config, "num_hash_layers", None) is None:
        config.num_hash_layers = 0
    return config


def _is_kda_layer(config, layer_idx: int) -> bool:
    """True if decoder layer `layer_idx` is a KDA linear-attention layer."""
    config = _text_config(config)
    layer_types = getattr(config, "layer_types", None)
    if layer_types is None:
        # Fallback: full-attention layers listed in linear_attn_config.
        full = set(
            getattr(config.linear_attn_config, "full_attn_layers", []) or [])
        return layer_idx not in full
    return layer_types[layer_idx] == "linear_attention"


class Glm5NextKDAAttention(VllmKimiGatedDeltaNetAttention):
    """KDA (Kimi Delta Attention) block for GLM-5.3-Flash.

    Subclasses the tpu_inference ``VllmKimiGatedDeltaNetAttention`` so the
    delta-rule core runs through the JAX GDN kernel
    (``run_jax_gdn_attention``). GLM-5.3's ``linear_attn_config`` uses the
    same keys (num_heads, head_dim, short_conv_kernel_size, gate_lower_bound)
    as vLLM's KimiLinearConfig, and the checkpoint stores the same parameter
    names (f_a_proj/f_b_proj/dt_bias/A_log, b_proj, g_a_proj/g_b_proj,
    q/k/v proj + conv1d, o_norm/o_proj).
    """

    pass


class Glm5NextMLAAttention(nn.Module):
    """MLA (+ optionally DSA) attention block for GLM-5.3-Flash.

    Delegates to upstream vLLM's ``DeepseekV2MLAAttention``; on TPU its
    internal ``mla_attn`` is the Pallas MLA implementation registered via
    register_oot. GLM-5.3 uses ``qk_rope_head_dim == 0`` ("pure nope" MLA,
    ``mla_use_nope=True``) with interleaved indexer RoPE applied inside the
    DSA indexer only, so the dense MLA path needs no positional encoding.

    The DSA lightning indexer is disabled while ``GLM_DSA_DISABLE=1`` so the
    MLA backbone runs dense (full-context) as a bring-up milestone.
    """

    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = _text_config(vllm_config.model_config.hf_config)
        self.layer_idx = extract_layer_index(prefix)

        self.mla = _UpstreamDeepseekV2MLAAttention(
            vllm_config=vllm_config,
            config=config,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            q_lora_rank=config.q_lora_rank,
            kv_lora_rank=config.kv_lora_rank,
            max_position_embeddings=config.max_position_embeddings,
            cache_config=vllm_config.cache_config,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.attn_mha",
        )
        if envs.GLM_DSA_DISABLE:
            self._disable_dsa_indexer()

    def _disable_dsa_indexer(self) -> None:
        """Force this MLA layer to run dense by removing its DSA indexer."""
        for attn in (self.mla, getattr(self.mla, "mla_attn", None)):
            if attn is None:
                continue
            if getattr(attn, "indexer", None) is not None:
                attn.indexer = None
            if hasattr(attn, "skip_topk"):
                attn.skip_topk = True

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.mla(positions, hidden_states)


class Glm5NextDecoderLayer(nn.Module):
    """GLM-5.3-Flash decoder layer: mHC -> (KDA | MLA) -> mHC -> (MoE | MLP).

    Structurally identical to DeepSeek-V4's decoder layer (same mHC
    parameterization and fused post/pre op); only the attention block differs
    per layer type and the FFN is dense for the first
    ``first_k_dense_replace`` layers.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()

        # Registers torch.ops.vllm.mhc_pre / mhc_post (lazy tilelang import).
        import vllm.model_executor.layers.mhc  # noqa: F401

        config = _text_config(vllm_config.model_config.hf_config)
        self.hidden_size = config.hidden_size
        self.rms_norm_eps = config.rms_norm_eps
        self.layer_idx = extract_layer_index(prefix)

        # Sub-blocks (KDA attention, DeepseekV2MLAAttention, DeepseekV4MoE)
        # read LM attrs from vllm_config.model_config.hf_config; expose the
        # unwrapped text config to them via a shallow proxy.
        import copy
        sub_model_config = copy.copy(vllm_config.model_config)
        sub_model_config.hf_config = config
        sub_vllm_config = copy.copy(vllm_config)
        sub_vllm_config.model_config = sub_model_config

        if _is_kda_layer(config, self.layer_idx):
            self.attn = Glm5NextKDAAttention(
                config, sub_vllm_config, prefix=f"{prefix}.attn")
        else:
            self.attn = Glm5NextMLAAttention(
                sub_vllm_config, prefix=f"{prefix}.attn")

        if self.layer_idx < config.first_k_dense_replace:
            self.ffn = DeepseekV4MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                swiglu_limit=getattr(config, "swiglu_limit", None),
                quant_config=vllm_config.quant_config,
                prefix=f"{prefix}.ffn",
            )
        else:
            self.ffn = DeepseekV4MoE(sub_vllm_config, prefix=f"{prefix}.ffn")

        self.attn_norm = RMSNorm(self.hidden_size, self.rms_norm_eps)
        self.ffn_norm = RMSNorm(self.hidden_size, self.rms_norm_eps)

        self.hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.hc_post_alpha = 2.0
        mix_hc = (2 + self.hc_mult) * self.hc_mult
        hc_dim = self.hc_mult * self.hidden_size
        self.hc_attn_fn = nn.Parameter(
            torch.empty((mix_hc, hc_dim), dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_ffn_fn = nn.Parameter(
            torch.empty((mix_hc, hc_dim), dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_attn_base = nn.Parameter(
            torch.empty(mix_hc, dtype=torch.float32), requires_grad=False)
        self.hc_ffn_base = nn.Parameter(
            torch.empty(mix_hc, dtype=torch.float32), requires_grad=False)
        self.hc_attn_scale = nn.Parameter(
            torch.empty(3, dtype=torch.float32), requires_grad=False)
        self.hc_ffn_scale = nn.Parameter(
            torch.empty(3, dtype=torch.float32), requires_grad=False)
        self.mhc_pre = MHCPreOp()
        self.mhc_post = MHCPostOp()
        self.mhc_fused_post_pre = MHCFusedPostPreOp()

    def hc_pre(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
    ):
        post_mix, res_mix, layer_input = self.mhc_pre(
            residual=x,
            fn=hc_fn,
            hc_scale=hc_scale,
            hc_base=hc_base,
            rms_eps=self.rms_norm_eps,
            hc_pre_eps=self.hc_eps,
            hc_sinkhorn_eps=self.hc_eps,
            hc_post_mult_value=self.hc_post_alpha,
            sinkhorn_repeat=self.hc_sinkhorn_iters,
        )
        return layer_input, post_mix, res_mix

    def hc_post(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ):
        return self.mhc_post(x, residual, post, comb)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        # Unfused mHC bring-up path (mirrors DeepseekV4DecoderLayer).
        residual = x
        x, post, comb = self.hc_pre(x, self.hc_attn_fn, self.hc_attn_scale,
                                    self.hc_attn_base)
        x = self.attn_norm(x)
        if isinstance(self.attn, Glm5NextMLAAttention):
            x = self.attn(positions, x)
        else:
            # TPU KDA forward signature: (hidden_states, output).
            x = self.attn(x, None)
        x = self.hc_post(x, residual, post, comb)

        residual = x
        x, post, comb = self.hc_pre(x, self.hc_ffn_fn, self.hc_ffn_scale,
                                    self.hc_ffn_base)
        x = self.ffn_norm(x)
        if isinstance(self.ffn, DeepseekV4MoE):
            x = self.ffn(x, input_ids)
        else:
            x = self.ffn(x)
        x = self.hc_post(x, residual, post, comb)
        return x


class Glm5NextTextModel(nn.Module):
    """GLM-5.3-Flash text backbone (embedding + 45 mHC decoder layers)."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = _text_config(vllm_config.model_config.hf_config)
        self.config = config
        self.vocab_size = config.vocab_size
        self.hc_mult = config.hc_mult
        self.hc_dim = self.hc_mult * config.hidden_size

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: Glm5NextDecoderLayer(vllm_config, prefix),
            prefix=maybe_prefix(prefix, "layers"),
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.hc_head_op = HCHeadOp()

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if intermediate_tensors is not None:
            # Pipeline-parallel: not a bring-up target on single-host v6e.
            raise NotImplementedError(
                "Glm5Next TPU bring-up does not support pipeline parallelism.")

        if inputs_embeds is None:
            hidden_states = self.get_input_embeddings(input_ids)
        else:
            hidden_states = inputs_embeds

        batch_size = hidden_states.shape[0]
        # Expand token embeddings into hc_mult residual streams: the mHC
        # residual is carried as (num_tokens, hc_mult * hidden_size).
        residual = hidden_states.unsqueeze(1).expand(
            batch_size, self.hc_mult, self.config.hidden_size).reshape(
                batch_size, self.hc_dim).contiguous()

        x = residual
        for layer in self.layers[self.start_layer:self.end_layer]:
            if isinstance(layer, PPMissingLayer):
                continue
            x = layer(x, positions, input_ids)

        # Collapse the hc_mult streams back to (num_tokens, hidden_size).
        x = self.hc_head_op(x, self.hc_mult)
        return self.norm(x)


class Glm5NextTextForCausalLM(nn.Module, SupportsPP):
    """GLM-5.3-Flash text-only causal LM for TPU (torchax).

    Handles the HF checkpoint layout of zai-org/GLM-5.3-Flash
    (``model.language_model.*``) including fp8 block-wise scales
    (``weight_scale_inv``) and the MTP next-layer tensors
    (``layers.45.*``, ``shared_head.*``), which are skipped by the base model.
    """

    packed_modules_mapping = {
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = _text_config(vllm_config.model_config.hf_config)
        self.config = config
        self.quant_config = vllm_config.quant_config

        self.model = Glm5NextTextModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=self.quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)

        logger.info(
            "Initialized Glm5NextTextForCausalLM (GLM-5.3-Flash) on TPU "
            "torchax path: num_hidden_layers=%d, kda_layers=%d, "
            "n_routed_experts=%s, MLA kv_lora_rank=%s, hc_mult=%s, "
            "GLM_DSA_DISABLE=%s",
            config.num_hidden_layers,
            sum(1 for i in range(config.num_hidden_layers)
                if _is_kda_layer(config, i)),
            getattr(config, "n_routed_experts", None),
            getattr(config, "kv_lora_rank", None),
            getattr(config, "hc_mult", None),
            envs.GLM_DSA_DISABLE,
        )

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        # vLLM's VllmModel protocol entrypoint.
        return self.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, intermediate_tensors,
                          inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str,
                                                     torch.Tensor]]) -> None:
        """Load GLM-5.3-Flash checkpoint weights.

        The HF checkpoint nests the text model under
        ``model.language_model.`` and stores per-expert weights as separate
        tensors (``mlp.experts.<e>.<proj>.weight``); vLLM's fused modules use
        ``ffn.*`` names. We therefore:

        1. Strip the ``model.language_model.`` prefix.
        2. Rename HF module names to the vLLM layout used by this port
           (``self_attn -> attn.attn_mha`` for MLA, ``mlp -> ffn``,
           KDA q/k/v proj + conv1d -> fused in_proj_qkvgfab / conv1d handled
           by KimiGatedDeltaNetAttention's own loaders).
        3. Skip vision (``model.visual.``), MTP (``layers.45.``,
           ``shared_head.``, ``eh_proj``, ``enorm``, ``hnorm``) tensors.
        """

        # Dense-MLP and MoE shared-experts fuse gate_proj+up_proj into
        # gate_up_proj; the AutoWeightsLoader does not apply
        # packed_modules_mapping for these, so we route them explicitly with
        # shard ids. Everything else is name-translated and delegated.
        gate_up: dict[str, dict[str, torch.Tensor]] = {}
        passthrough: list[tuple[str, torch.Tensor]] = []
        for name, weight in weights:
            tname = self._translate_weight_name(name)
            if tname is None:
                continue
            if ".gate_proj." in tname or ".up_proj." in tname:
                # Skip MoE routed experts (mlp.experts.*) — FusedMoEFactory
                # handles those; only fuse the dense MLP / shared experts.
                if ".experts." not in tname:
                    is_scale = tname.endswith("weight_scale_inv")
                    base = tname.replace(".gate_proj.", ".gate_up_proj.") \
                                .replace(".up_proj.", ".gate_up_proj.")
                    shard = "gate" if ".gate_proj." in tname else "up"
                    key = f"{shard}_scale" if is_scale else shard
                    gate_up.setdefault(base, {})[key] = weight
                    continue
            passthrough.append((tname, weight))

        params = dict(self.named_parameters())
        for base, shards in gate_up.items():
            for shard in ("gate", "up"):
                if shard in shards:
                    param = params[base]
                    loader_fn = getattr(param, "weight_loader",
                                        default_weight_loader)
                    loader_fn(param, shards[shard], shard)
                scale_key = f"{shard}_scale"
                if scale_key in shards:
                    # weight_scale_inv has no shard loader; both gate and up
                    # scale tensors land in the same fused scale buffer.
                    scale_param = params[base]
                    default_weight_loader(scale_param, shards[scale_key])

        loader = AutoWeightsLoader(
            self,
            ignore_unexpected_prefixes=[
                "model.visual.",
                "visual.",
                "model.layers.45.",
            ],
        )
        loader.load_weights(iter(passthrough))

    # ------------------------------------------------------------------
    # Weight-name translation (HF checkpoint -> this vLLM port)
    # ------------------------------------------------------------------

    def _translate_weight_name(self, name: str) -> str | None:
        """Translate an HF checkpoint tensor name to this port's layout.

        Returns None for tensors that should be skipped (vision / MTP).
        """
        if name.startswith("model.language_model."):
            name = "model." + name[len("model.language_model."):]
        if name.startswith("model.visual.") or name.startswith("visual."):
            return None
        # MTP tensors are skipped entirely (routed to a skip prefix).
        if ".eh_proj." in name or ".enorm." in name or ".hnorm." in name \
                or ".shared_head." in name or ".mapping_proj." in name:
            return "model.layers.45." + name
        # The checkpoint stores MTP next-layer tensors as model.layers.45.*;
        # the base model only has layers 0..44, so skip them before any
        # layer-type lookup (which would index out of range).
        if name.startswith("model.layers.45."):
            return None

        # KDA layers store the block as `attn` (same as MLA); only the MLA
        # submodule names (q_a/q_b/kv_a/kv_b + layernorms + indexer) map to
        # the upstream `attn.attn_mha` nesting. KDA names pass through.
        is_mla = any(k in name for k in (
            "q_a_proj", "q_b_proj", "kv_a_proj", "kv_b_proj", "q_a_layernorm",
            "kv_a_layernorm", "indexer."))
        # o_proj/o_norm appear in both KDA and MLA; decide by layer type.
        if not is_mla and (".self_attn.o_proj." in name or
                           ".self_attn.o_norm." in name):
            layer_idx = int(name.split("model.layers.")[1].split(".", 1)[0])
            is_mla = not _is_kda_layer(self.config, layer_idx)
        if is_mla:
            name = name.replace("self_attn.", "attn.attn_mha.")
        else:
            name = name.replace("self_attn.", "attn.")
        # MoE / MLP: mlp -> ffn.
        name = name.replace("mlp.", "ffn.")
        # Decoder-layer norms use the DeepseekV4 attribute names in this port.
        name = name.replace("input_layernorm.", "attn_norm.")
        name = name.replace("post_attention_layernorm.", "ffn_norm.")
        return name
