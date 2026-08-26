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
(separate ``q/k/v_proj`` + ``q/k/v_conv1d``, low-rank forget gate
``f_a -> f_b`` producing a per-head scalar gate, low-rank output gate
``g_a -> g_b``). Only the delta-rule core differs: on TPU it runs through the
JAX GDN kernel (``run_jax_gdn_attention``) instead of vLLM's
``torch.ops.vllm.kda_attention`` (Triton), reusing the same
``gdn_attention_core_tpu`` bridge as Qwen GDN.
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

    Mirrors upstream vLLM's ``KimiGatedDeltaNetAttention`` for the projection
    and output-gate/norm; the conv + chunked/recurrent delta-rule core is
    delegated to ``gdn_attention_core_tpu`` so it runs on the TPU JAX GDN
    kernel. The JAX kernel expects a fused ``mixed_qkv`` (num_tokens,
    3*num_heads*head_dim), a fused conv weight, and per-head beta/gate.
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

    @property
    def conv1d(self):
        """Fused (q, k, v) conv1d view expected by the TPU GDN core.

        The JAX kernel consumes a single conv weight of shape
        ``(3*num_heads*head_dim, 1, kernel_size)``; this build keeps them as
        three separate ``{q,k,v}_conv1d`` Linears, so we fuse lazily.
        """
        if getattr(self, "_fused_conv1d_weight", None) is None:
            w = torch.cat(
                [
                    self.q_conv1d.weight,
                    self.k_conv1d.weight,
                    self.v_conv1d.weight,
                ],
                dim=0,
            )
            self._fused_conv1d_weight = w

        class _Conv:
            weight = self._fused_conv1d_weight
            bias = None

        return _Conv()

    def forward(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """KDA forward with the delta-rule core on the TPU JAX kernel.

        Args:
            hidden_states: ``(num_tokens, hidden_size)``.
            output: Preallocated output buffer (num_tokens, hidden_size).

        Returns:
            ``(num_tokens, hidden_size)`` attention output.
        """
        vllm_model_wrapper_context = get_vllm_model_wrapper_context()
        mesh = vllm_model_wrapper_context.mesh
        num_tokens = hidden_states.size(0)

        # ---------------- Part 1: input projection ----------------
        q, _ = self.q_proj(hidden_states)
        k, _ = self.k_proj(hidden_states)
        v, _ = self.v_proj(hidden_states)
        # Fused (num_tokens, 3 * num_heads * head_dim), matching the
        # non-interleaved [Q Q | K K | V V] layout the JAX GDN core expects.
        mixed_qkv = torch.cat([q, k, v], dim=-1)

        # Forget gate: low-rank f_a -> f_b -> per-head scalar gate g1.
        f_a, _ = self.f_a_proj(hidden_states)
        g1, _ = self.f_b_proj(f_a)
        # Beta gate.
        beta, _ = self.b_proj(hidden_states)
        # Output gate (low-rank g_a -> g_b).
        g_proj_states, _ = self.g_b_proj(self.g_a_proj(hidden_states)[0])
        g2 = rearrange(g_proj_states, "... (h d) -> ... h d", d=self.head_dim)

        # ---------------- Part 2: core attention (TPU kernel) ----------------
        # beta and g1 are per-head (num_tokens, num_heads).
        b = beta.contiguous().view(num_tokens, -1)
        a = g1.contiguous().view(num_tokens, -1)

        core_attn_out = torch.zeros(
            (num_tokens, self.num_heads // self.tp_size, self.head_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        gdn_attention_core_tpu(mixed_qkv, b, a, core_attn_out, self.prefix,
                               mesh=mesh)

        # ---------------- Part 3: gated norm + output projection ----------------
        core_attn_out = self.o_norm(core_attn_out, g2)
        core_attn_out = rearrange(core_attn_out, "... h d -> ... (h d)")
        proj_out, _ = self.o_proj(core_attn_out)
        if output is not None:
            output[:num_tokens] = proj_out
            return output
        return proj_out
