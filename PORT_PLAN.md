# Port Plan: Qwen/Qwen3.8-27B on TPU v6e (tpu-inference)

## Target model architecture (from HF config.json)
- `architectures`: **Qwen3_5ForConditionalGeneration** (multimodal; `language_model_only: false`)
- `model_type`: `qwen3_5` — the **Qwen3.5 hybrid** family (NOT dense Qwen3). Deltas vs Qwen3:
  - **Hybrid attention**: 64 layers, pattern = 3× `linear_attention` + 1× `full_attention` (`full_attention_interval: 4`).
    - Linear layers = **Gated DeltaNet** (GDN): `linear_num_key_heads: 16`, `linear_num_value_heads: 48`,
      `linear_key_head_dim: 128`, `linear_value_head_dim: 128`, `linear_conv_kernel_dim: 4`,
      `mamba_ssm_dtype: float32` — stateful SSM-like recurrence (conv state + recurrent state), no KV cache for these layers.
    - Full-attention layers: GQA, `num_attention_heads: 24`, `num_key_value_heads: 4`, `head_dim: 256`,
      `attn_output_gate: true` (swish output gate), `partial_rotary_factor: 0.25`, `rope_theta: 1e7`,
      mrope sections `[11, 11, 10]` (`mrope_interleaved: true`).
  - Dense MLP (not MoE): `hidden_size: 5120`, `intermediate_size: 17408`, SiLU. (Sibling 397B-A17B is the MoE variant.)
  - `vocab_size: 248320`, `tie_word_embeddings: false`, `rms_norm_eps: 1e-6`, bf16 weights.
  - Multimodal: separate `vision_config` (27-layer ViT, out 5120). We run **text-only** via `--language-model-only`.
- Weight size: ~27B params bf16 ≈ 54 GB (+ vision tower a few GB).

## Hardware / parallelism
- VM: `node-1`, europe-west4-a, **v6e-4** = 4 chips × 32 GB HBM = 128 GB total.
- bf16 54 GB weights ⇒ **TP=4** (weights ~13.5 GB/chip, leaves room for KV cache + GDN states). TP=2 impossible.
- Only full-attention layers (16 of 64) need KV cache; GDN layers keep fixed-size recurrent/conv state per seq.

## Porting path decision → **Case A/C hybrid: existing repo support, torchax path**
The local repo (main @ c81e2c79, newer than upstream) ALREADY supports Qwen3.5 on TPU:
- GDN kernels: `tpu_inference/kernels/gdn/{reference,v1,v2}` (+ tests `tests/kernels/*gdn*`).
- CI/e2e configs: `.buildkite/models/Qwen_Qwen3_5-{9B,397B-A17B}.yml`,
  `tests/e2e/benchmarking/inferencex/qwen3.5/server.sh`.
- `Qwen3_5MoeForConditionalGeneration` is in `_VLLM_PREFERRED_ARCHITECTURES` (runs via
  `MODEL_IMPL_TYPE=vllm` = torchax PyTorch path). No `flax_nnx` JAX model for Qwen3_5 exists yet.
- vLLM pin `.buildkite/vllm_lkg.version` = a7a204cc (2026-07-30) **contains** `vllm/model_executor/models/qwen3_5.py`.

Therefore: **no new model class needed initially.** Run the dense 27B with `MODEL_IMPL_TYPE=vllm`,
mirroring `tests/e2e/benchmarking/inferencex/qwen3.5/server.sh` but adapted for dense (drop MoE/EP flags)
and TP=4. `MODEL_IMPL_TYPE=auto` also falls back to the vLLM path for unregistered archs.

## Key env/flags (adapted from CI server.sh for dense TP4)
- `MODEL_IMPL_TYPE=vllm`, `NEW_MODEL_DESIGN=1`, `SLICE_ROPE_CACHE=1`
- `RAGGED_GATED_DELTA_RULE_IMPL=chunked_kernel_p_recurrent_kernel_d` (prefill chunked kernel, decode recurrent kernel)
- `--language-model-only --limit-mm-per-prompt '{"image":0,"video":0}'`
- `--tensor-parallel-size 4`, conservative `--max-model-len` (start 1024 offline / 4096 serve), `--block-size 256`
- NO MoE flags (`--enable-expert-parallel`, MOE_* envs) — this checkpoint is dense.
- HF transfer: no HF_TOKEN locally; model appears public (config raw = 200). Fallback: `gcloud secrets` on VM.

## VM setup (REPROVISIONED — original node-1 had unsupported legacy image)
- **Blocker found**: original `node-1` ran `tpu-vm-base` (2022, kernel 5.13) which does NOT support v6e
  Trillium → libtpu 0.0.44 could not enumerate chips (`Failed to get global TPU topology`). Abandoned.
- **New VM** (user-directed): queued resource `shivaji-qr-node-1`, node `shivaji-cust-tpu-vm-inf`,
  v6e-4, europe-west4-a, image **`v2-alpha-tpuv6e`** (Trillium-specific, PyTorch/XLA preinstalled).
  SSH: `gcloud compute tpus tpu-vm ssh shivaji-cust-tpu-vm-inf --zone europe-west4-a --project northam-ce-mlai-tpu`.
- v2-alpha-tpuv6e preinstalls torch/xla + libtpu matched to v6e. Still need: vllm @ a7a204cc
  (VLLM_TARGET_DEVICE=tpu) + editable tpu-inference. gcc-11 may still be needed for the spinloop C++ build.
- Env: `HF_HOME=/dev/shm/hf_cache` (RAM-backed; confirm VM RAM >= ~60 GB) to avoid disk pressure.

## Follow-up goal (user-requested, after torchax works)
- **JAX-native (flax_nnx) port** of Qwen3_5: new model class(es) in `tpu_inference/models/jax/`
  (e.g. `qwen3_5.py`) wiring the existing GDN kernels + full-attention layers, registered in
  `_MODEL_REGISTRY` for `Qwen3_5ForConditionalGeneration` / dense text-only. Larger effort; torchax
  path is the first milestone.

## Files to create/modify
- [x] `PORT_PLAN.md` (this file)
- [ ] Possibly none if Case A works out of the box.
- [ ] Likely fixes: config-field handling for dense `qwen3_5` (CI only exercised MoE 397B + 9B),
      GDN kernel numerics/shape issues at our dims (kv heads 16/48, head_dim 128, conv 4),
      weight-name mapping / transposes, rope mrope for text-only.
- [ ] `PORT_REPORT.md` at the end.

## Verification (definition of done)
- Offline `examples/offline_inference.py` greedy T=0 on ≥5 diverse prompts → coherent text.
- `vllm serve` + `curl /v1/completions` → coherent.
- If garbage: check weight-mapping coverage logs, transposes/fusion order, rope/QK-norm,
  GDN state handling; compare layer-0 hidden states vs HF transformers on CPU for a 5-token prompt.

## Milestones
1. [ ] VM env built (python3.12 + vllm + tpu-inference editable), `jax.devices()` shows 4 TPU chips.
2. [ ] Weights downloaded (~54 GB).
3. [ ] Offline inference loads (compile OK) — first XLA compile may take 30–60 min; poll logs.
4. [ ] Coherent offline outputs.
5. [ ] `vllm serve` + curl coherent.
6. [ ] Branch `port/qwen3.8-27b` + `PORT_REPORT.md`.

## Open questions / risks
- Disk: 84 GB free vs ~54 GB weights + ~10 GB venv + HF temp → use /dev/shm for HF cache; confirm VM RAM.
- python3.12 via deadsnakes on Ubuntu 20.04 (needs apt PPA + network).
- Dense qwen3_5 path may be less exercised than MoE — expect weight-loader/config bugs.
- GDN recurrent state vs vLLM mamba-style cache manager integration on torchax path.
