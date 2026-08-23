# Port Report: Qwen/Qwen3.8-27B on TPU v6e (Trillium) via tpu-inference

**Status: ✅ DONE & VERIFIED.** `vllm serve Qwen/Qwen3.8-27B` and the offline example both run on a
v6e-4 TPU VM and produce coherent, factually-correct text across a diverse prompt suite.

## 1. Model architecture summary
`Qwen/Qwen3.8-27B` → HF `architectures: ["Qwen3_5ForConditionalGeneration"]`, `model_type: qwen3_5`.
Despite the "Qwen3.8" name, it is the **Qwen3.5 hybrid** family, **not** a dense Qwen3 clone:

- **Hybrid attention, 64 layers**: repeating `[linear_attention ×3, full_attention ×1]` (`full_attention_interval: 4`).
  - 48 **Gated DeltaNet (GDN)** linear layers: `linear_num_key_heads=16`, `linear_num_value_heads=48`,
    `linear_key_head_dim=128`, `linear_value_head_dim=128`, `linear_conv_kernel_dim=4`,
    `mamba_ssm_dtype=float32`. These keep a fixed-size recurrent+conv state per sequence (no KV cache).
  - 16 **full-attention** layers: GQA `num_attention_heads=24`, `num_key_value_heads=4`, `head_dim=256`,
    `attn_output_gate=true` (swish output gate), `partial_rotary_factor=0.25`, `rope_theta=1e7`,
    mrope sections `[11,11,10]`.
- **Dense MLP** (not MoE): `hidden_size=5120`, `intermediate_size=17408`, SiLU.
- `vocab_size=248320`, `tie_word_embeddings=false`, `rms_norm_eps=1e-6`, bf16.
- **Multimodal** (vision tower, 27-layer ViT) — run **text-only** via `--language-model-only`.
- Weights: 18 safetensors shards, **55.6 GB** (public repo).

## 2. Porting path — Case A/C: existing support, torchax (no new model class needed)
The local `tpu-inference` repo already supports Qwen3.5 on TPU:
- GDN TPU kernels: `tpu_inference/kernels/gdn/{reference,v1,v2}`.
- torchax GDN op: `tpu_inference/layers/vllm/custom_ops/gdn_attention_op.py` (`VllmGatedDeltaNetAttention`,
  registered out-of-tree).
- CI/e2e references: `.buildkite/models/Qwen_Qwen3_5-*.yml`, `tests/e2e/benchmarking/inferencex/qwen3.5/server.sh`.
- Pinned vLLM (`.buildkite/vllm_lkg.version` = `a7a204cc`, 2026-07-30) ships `vllm/model_executor/models/qwen3_5.py`.

So we run the dense 27B with the **torchax** path (`MODEL_IMPL_TYPE=vllm`), which executes the vLLM
PyTorch `qwen3_5` definition on TPU and dispatches GDN/full-attention to the TPU kernels.
`MODEL_IMPL_TYPE=auto` also resolves to this for unregistered archs.

## 3. Environment / infra (important — one real blocker)
- **Blocker:** the original `node-1` ran the legacy `tpu-vm-base` (2022, kernel 5.13) image, which
  **does not support v6e Trillium** → libtpu 0.0.44 could not enumerate chips
  (`RuntimeError: Unable to initialize backend 'tpu': INTERNAL: Failed to get global TPU topology`).
  No Python-level fix exists; the image/kernel is the problem.
- **Fix:** recreated the TPU as a **queued resource** with a Trillium image:
  ```
  gcloud compute tpus queued-resources create shivaji-qr-node-1 \
    --node-id shivaji-cust-tpu-vm-inf --project northam-ce-mlai-tpu \
    --zone europe-west4-a --accelerator-type v6e-4 --runtime-version v2-alpha-tpuv6e
  ```
  New node `shivaji-cust-tpu-vm-inf`: Ubuntu 22.04, kernel 6.8, 708 GB RAM, 355 GB /dev/shm, 4× v6e chips.

### Build steps on the VM (reproducible)
```
# uv + python 3.12 venv
curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"
uv python install 3.12 && uv venv ~/vllm_env --python 3.12 && source ~/vllm_env/bin/activate
# vllm at the repo-pinned commit
git clone --filter=blob:none --no-checkout https://github.com/vllm-project/vllm.git ~/vllm
cd ~/vllm && git fetch --depth 1 origin a7a204cc6ec99b375985e564f4f271e68ee8f5b8 && git checkout a7a204cc
export UV_TORCH_BACKEND=cpu
uv pip install -r requirements/tpu.txt
VLLM_TARGET_DEVICE=tpu uv pip install -e . --no-build-isolation   # needs gcc/g++ >= 11.3 (22.04 has 11.4)
# tpu-inference (synced repo), editable
cd ~/tpu-inference && uv pip install -e .
# sanity: python -c "import jax; print(len(jax.devices()))"  -> 4
```
Weights cached to RAM to avoid disk pressure: `export HF_HOME=/dev/shm/hf_cache`.

## 4. Run commands
### Offline (first milestone)
```
export HF_HOME=/dev/shm/hf_cache MODEL_IMPL_TYPE=vllm NEW_MODEL_DESIGN=1 SLICE_ROPE_CACHE=1 \
  RAGGED_GATED_DELTA_RULE_IMPL=chunked_kernel_p_recurrent_kernel_d
python examples/offline_inference.py --model Qwen/Qwen3.8-27B \
  --tensor-parallel-size 4 --max-model-len 1024 --max-num-seqs 4 \
  --language-model-only --temperature 0 --max-tokens 64 --gpu-memory-utilization 0.9
```
### Serve (definition-of-done)
```
vllm serve Qwen/Qwen3.8-27B --tensor-parallel-size 4 --max-model-len 4096 --max-num-seqs 8 \
  --language-model-only --gpu-memory-utilization 0.9 --port 8000   # same env vars as above
curl localhost:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3.8-27B","prompt":"The capital of France is","max_tokens":32,"temperature":0}'
```

## 5. Code change (the only one needed) — branch `port/qwen3.8-27b`
`tpu_inference/models/vllm/vllm_model_wrapper.py` — `wrap_precompile_vision_encoder_fn`:
**Bug:** it precompiled the vision encoder whenever the model was multimodal, **without** checking
`language_model_only`. Under text-only mode the vision tower is a `StageMissingLayer` stub, so the
serve-path AOT precompile crashed with `RuntimeError: StageMissingLayer(stage_name='vision_tower')
should not be called`. (The offline `LLM` path skipped this, which is why offline worked first.)
**Fix:** return `None` (skip vision precompile) when `model_config.multimodal_config.language_model_only`
is true. One guarded early-return; no behavior change for true multimodal serving.

## 6. Verification (greedy, temperature 0)
Offline: all 12 default prompts coherent & correct. Serve (`/v1/completions`), 10 diverse prompts — all correct:
- factual: "The capital of France is" → "Paris…"; "chemical symbol for gold" → "Au (aurum), Z=79".
- math: "17 × 24" → "408"; "2x+6=20" → "x=7"; "60 mph × 2.5 h" → "150 miles".
- code: Fibonacci function (valid, 0-indexed w/ validation); list-comprehension completion `[x*x for x in range(10) if x % 2 == 0]` → `[0,4,16,36,64]`.
- multilingual: "Good morning" → French "Bonjour, comment allez-vous ?", German "Guten Morgen…".
- knowledge/reasoning: Romeo & Juliet → Shakespeare, 16th century (~1595–96); blue sky → Rayleigh scattering.

## 7. Performance (v6e-4, TP=4, bf16, no tuning)
- Memory: weights **12.65 GiB/chip** (~50.6 GiB total). HBM used 50.61 GiB; avail for KV/state 61.86 GiB.
- Hybrid cache (serve, max_model_len 4096): **16 attn layers** KV = 60.56 GiB; **48 GDN layers** state = 1.29 GiB.
  GDN state shape per layer: conv `(9,3,10240)` bf16 + recurrent `(9,48,128,128)` fp32.
- Throughput (single-stream, from server logs): generation up to **~44 tokens/s**; prefill prompt
  throughput scales with length. First-time XLA compile dominates startup (several minutes, many shape buckets).
- Cold start: weight load ~2 s (18 shards from RAM cache); engine ready a few minutes after compile.

## 8. Known limitations / TODO
- **Text-only**: vision tower disabled (`--language-model-only`). Multimodal (image/video) serving is
  untested and would need the vision precompile path (and the above guard removed for that mode).
- **TP must be 4** on v6e-4 (54 GB bf16 can't fit < 4 chips with KV headroom). TP=2 impossible.
- **JAX-native (flax_nnx) port**: user-requested follow-up. Add `tpu_inference/models/jax/qwen3_5.py`
  (dense, text-only) wiring the existing GDN kernels + full-attention layers, and register
  `Qwen3_5ForConditionalGeneration` in `_MODEL_REGISTRY`. Not required for the torchax DoD above.
- MoE sibling (`Qwen3.5-397B-A17B`) uses the same path plus expert-parallel flags; not exercised here.
- Minor: `tpu_inference` editable install reports version `0.0.0`; the `libtpu` THP "file offset
  incorrectly aligned" message is a harmless warning on this image.

## 9. Files changed
- `tpu_inference/models/vllm/vllm_model_wrapper.py` (bugfix, §5)
- `PORT_PLAN.md`, `PORT_REPORT.md` (docs)
