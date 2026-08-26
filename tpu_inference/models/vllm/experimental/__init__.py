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

# Mapping of architecture name -> "module:ClassName" for the TPU-specific
# vLLM/torchax model implementations that should override vLLM's built-in
# ones. Values use the lazy string form so the (torch/tilelang) model module
# is only imported when the architecture is actually instantiated.
_TPU_VLLM_MODELS = {
    "DeepseekV4ForCausalLM":
    "tpu_inference.models.vllm.experimental.deepseek_v4:DeepseekV4ForCausalLM",
    # GLM-5.2 (zai-org/GLM-5.2): MoE + MLA + DeepSeek Sparse Attention.
    # Subclasses upstream vLLM's GlmMoeDsaForCausalLM; the MLA attention is
    # routed to the TPU PallasMLA backend by the register_oot MLA wrapper.
    "GlmMoeDsaForCausalLM":
    "tpu_inference.models.vllm.experimental.glm_moe_dsa:GlmMoeDsaForCausalLM",
    # GLM-5.3-Flash (zai-org/GLM-5.3-Flash) text backbone: hybrid KDA linear
    # attention + MLA(+DSA) layers, mHC residuals, 288-expert MoE. Text-only
    # bring-up; the multimodal wrapper is a follow-up.
    "Glm5NextTextForCausalLM":
    "tpu_inference.models.vllm.experimental.glm5_next:Glm5NextTextForCausalLM",
    # The HF checkpoint advertises the multimodal wrapper architecture; map
    # it to the same text-only bring-up (vision tower not yet on TPU).
    "Glm5NextForConditionalGeneration":
    "tpu_inference.models.vllm.experimental.glm5_next:Glm5NextTextForCausalLM",
}


def register_models():
    """Override vLLM's built-in model classes with TPU-specific ones.

    Called from the ``vllm.general_plugins`` entrypoint so it runs at vLLM
    startup, before any model is resolved from its architecture name.
    """
    from vllm.model_executor.models.registry import (ModelRegistry,
                                                     _EMBEDDING_MODELS)

    for arch, model_cls in _TPU_VLLM_MODELS.items():
        ModelRegistry.register_model(arch, model_cls)

    # Register JAX pooling models for TPU inference
    from tpu_inference.models.jax.jina_bert import (
        JinaBertForMaskedLM,
        JinaBertModel,
    )
    from tpu_inference.models.common.model_loader import register_model
    register_model("JinaBertForMaskedLM", JinaBertModel)
    register_model("JinaBertModel", JinaBertModel)
    _EMBEDDING_MODELS["JinaBertForMaskedLM"] = ("jina", "JinaBertForMaskedLM")
    _EMBEDDING_MODELS["JinaBertModel"] = ("jina", "JinaBertModel")
