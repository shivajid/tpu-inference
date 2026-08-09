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
"""TPU (torchax) entrypoint for GLM-5.2 (``glm_moe_dsa``).

GLM-5.2 is a DeepSeek-V3.2-style model: a mixture-of-experts transformer with
Multi-head Latent Attention (MLA) and DeepSeek Sparse Attention (DSA, a
"lightning indexer" that selects the top-k KV tokens per query).

The heavy lifting on TPU is already handled by generic, model-agnostic
machinery:

* MoE / embeddings / router / output head run through vLLM's standard
  torchax-compatible modules.
* The MLA attention is swapped to the TPU Pallas MLA backend by
  ``VllmMultiHeadLatentAttentionWrapper`` (registered via ``register_oot``),
  which ``DeepseekV2MLAAttention`` builds internally.

This module subclasses the upstream implementation purely so that the
architecture name ``GlmMoeDsaForCausalLM`` resolves through the TPU torchax
model path. DSA (sparse top-k attention) is brought up incrementally; the MLA
backbone runs dense on TPU first while the indexer / sparse kernels are wired.
"""

from vllm.model_executor.models.deepseek_v2 import (
    GlmMoeDsaForCausalLM as _UpstreamGlmMoeDsaForCausalLM)

from tpu_inference import envs
from tpu_inference.logger import init_logger

logger = init_logger(__name__)


class GlmMoeDsaForCausalLM(_UpstreamGlmMoeDsaForCausalLM):
    """GLM-5.2 for TPU.

    Reuses upstream vLLM's ``GlmMoeDsaForCausalLM`` (MoE + MLA + DSA indexer).
    On TPU the MLA attention module is transparently replaced by the
    torchax-compatible Pallas MLA implementation. Weight loading, expert
    mapping, and the packed ``gate_up_proj`` / ``fused_qkv_a_proj`` mappings are
    inherited unchanged.

    Set ``GLM_DSA_DISABLE=1`` to bypass the DSA lightning indexer and run the
    MLA backbone dense (full-context) as a bring-up milestone before the TPU
    sparse top-k path is enabled.
    """

    def __init__(self, *, vllm_config, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        logger.info(
            "Initialized GlmMoeDsaForCausalLM (GLM-5.2) on TPU torchax path: "
            "num_hidden_layers=%d, n_routed_experts=%d, MLA kv_lora_rank=%s, "
            "DSA index_topk=%s",
            getattr(self.config, "num_hidden_layers", -1),
            getattr(self.config, "n_routed_experts", -1),
            getattr(self.config, "kv_lora_rank", None),
            getattr(self.config, "index_topk", None),
        )
        if envs.GLM_DSA_DISABLE:
            self._disable_dsa_indexer()

    def _disable_dsa_indexer(self) -> None:
        """Force every MLA layer to run dense by removing its DSA indexer."""
        disabled = 0
        for layer in self.model.layers:
            attn = getattr(layer, "self_attn", None)
            if attn is None or getattr(attn, "indexer", None) is None:
                continue
            attn.indexer = None
            attn.skip_topk = True
            mla_attn = getattr(attn, "mla_attn", None)
            if mla_attn is not None:
                if hasattr(mla_attn, "indexer"):
                    mla_attn.indexer = None
                if hasattr(mla_attn, "skip_topk"):
                    mla_attn.skip_topk = True
            disabled += 1
        logger.info(
            "GLM_DSA_DISABLE=1: disabled DSA indexer on %d MLA layers; "
            "running MLA dense (full-context).", disabled)
