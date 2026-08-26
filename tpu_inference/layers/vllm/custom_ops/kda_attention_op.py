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
"""TPU (torchax) bring-up of vLLM's Kimi KDA (Kimi Delta Attention) layer.

The KDA block used by GLM-5.3-Flash's ``linear_attention`` layers shares the
exact projection / gating structure of vLLM's ``KimiGatedDeltaNetAttention``
(fused ``in_proj_qkvgfab`` input projection, low-rank forget gate
``f_a -> f_b`` producing a per-head scalar gate, low-rank output gate
``g_a -> g_b``). Only the delta-rule core differs: on TPU it runs through the
JAX GDN kernel (``run_jax_gdn_attention``) instead of vLLM's Triton KDA
kernels, reusing the same ``gdn_attention_core_tpu`` bridge as Qwen GDN.
"""

import torch
from einops import rearrange
from vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn import (
    KimiGatedDeltaNetAttention)

from tpu_inference.layers.vllm.custom_ops.gdn_attention_op import (
    gdn_attention_core_tpu)
from tpu_inference.logger import init_logger
from tpu_inference.models.vllm.vllm_model_wrapper_context import (
    get_vllm_model_wrapper_context)

logger = init_logger(__name__)


@KimiGatedDeltaNetAttention.register_oot
class VllmKimiGatedDeltaNetAttention(KimiGatedDeltaNetAttention):
    """TPU bring-up of Kimi KDA (Kimi Delta Attention).

    Mirrors upstream vLLM's ``KimiGatedDeltaNetAttention`` for Part 1 (input
    projection) and Part 3 (gated norm + output projection); Part 2 (short
    conv1d + chunked/recurrent delta rule) is delegated to
    ``gdn_attention_core_tpu`` so it runs on the TPU JAX GDN kernel.
    """

    # Attribute aliases expected by ``gdn_attention_core_tpu`` (which reads
    # Qwen-style attribute names off the layer module).
    @property
    def num_k_heads(self) -> int:
        return self.num_heads

    @property
    def num_v_heads(self) -> int:
        return self.num_heads

    @property
    def head_k_dim(self) -> int:
        return self.head_dim

    @property
    def head_v_dim(self) -> int:
        return self.head_dim

    @property
    def conv_kernel_size(self) -> int:
        return self.conv_size

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """KDA forward with the delta-rule core on the TPU JAX kernel.

        Args:
            positions: Token positions (unused by the GDN/KDA core, which is
                position-independent; kept for interface parity).
            hidden_states: ``(num_tokens, hidden_size)``.
            output: Optional preallocated output buffer.

        Returns:
            ``(num_tokens, hidden_size)`` attention output.
        """
        vllm_model_wrapper_context = get_vllm_model_wrapper_context()
        mesh = vllm_model_wrapper_context.mesh
        num_tokens = hidden_states.size(0)

        # ---------------- Part 1: input projection ----------------
        projected_qkvgfab, _ = self.in_proj_qkvgfab(hidden_states)
        if self.use_full_rank_gate:
            split_sizes = [
                3 * self.local_projection_size,
                self.local_projection_size,
                self.head_dim,
                self.local_num_heads,
            ]
            if self.in_proj_padding:
                split_sizes.append(self.in_proj_padding)
            projected = projected_qkvgfab.split(split_sizes, dim=-1)
            mixed_qkv, g_proj_states, f_a, beta = projected[:4]
            g = g_proj_states
        else:
            mixed_qkv, beta, f_a = projected_qkvgfab.split(
                [
                    3 * self.local_projection_size,
                    self.local_num_heads,
                    self.head_dim,
                ],
                dim=-1,
            )
            g, _ = self.g_b_proj(self.g_a_proj(hidden_states)[0])

        # Forget gate: low-rank f_a -> f_b -> per-head scalar gate g1.
        g1, _ = self.f_b_proj(f_a)

        # ---------------- Part 2: core attention (TPU kernel) ----------------
        # The TPU GDN kernel expects beta (b) and the per-head gate (a) as
        # (num_tokens, num_v_heads); g1 is already per-head (num_tokens, 64).
        b = beta.contiguous().view(num_tokens, -1)
        a = g1.contiguous().view(num_tokens, -1)

        core_attn_out = torch.zeros(
            (num_tokens, self.num_heads // self.tp_size, self.head_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        gdn_attention_core_tpu(mixed_qkv, b, a, core_attn_out, self.prefix,
                               mesh=mesh)

        # ---------------- Part 3: output projection ----------------
        core_attn_out = self.o_norm(core_attn_out, g)
        core_attn_out = rearrange(core_attn_out, "... h d -> ... (h d)")
        if output is not None:
            output[:num_tokens], _ = self.o_proj(core_attn_out)
            return output
        return self.o_proj(core_attn_out)[0]
