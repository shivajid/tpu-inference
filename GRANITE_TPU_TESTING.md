# Testing Granite-3.3 on vLLM-TPU (tpu-inference)

This guide covers building, unit-testing, serving, numerically validating, and
benchmarking `ibm-granite/granite-3.3-8b-instruct` on the JAX-native
`GraniteForCausalLM` implementation in this branch.

## 0. Target environment

- **TPU host:** v6e-4 (4 chips, 32 GB HBM each). Access via
  `gcloud compute tpus tpu-vm ssh/scp --zone <ZONE> <HOST> --project <PROJECT>`.
- **Runtime:** Python **3.12** venv (tpu-inference pins `jax==0.11.0`, which
  requires py≥3.12; a py3.11 venv fails). jax 0.11.0 + libtpu.
- **vLLM:** built from source at the commit paired with this tpu-inference
  branch. Server reports e.g. `0.23.1rc1.dev...`.

> Host apt deps (already present on a provisioned host): `cmake` (≥3.26 via
> pip), `libopenmpi-dev`, `libomp-dev`, `build-essential`, `python3.12`
> (+ `-venv`, `-dev`). System cmake 3.22 is too old — put the venv's
> pip-installed cmake on `PATH`.

## 1. Build the runtime

```bash
# On the TPU host. Creates ~/vllm-env (py3.12) if missing, builds vLLM + tpu-inference.
~/build_tpu_inf.sh [vllm_commit]   # default vllm commit 34b560b
```

Notes:
- `pip` will show `tpu_inference 0.0.0` (editable, no `VLLM_VERSION_OVERRIDE`);
  a cosmetic "requires tpu-inference==X" warning on rebuild is harmless.
- Auto-register "cannot import name ..." warnings seen when importing
  `model_loader` in isolation are red herrings; the real flow (`import vllm`
  first) resolves Granite at serve time (`Resolved architecture:
  GraniteForCausalLM`).

## 2. Unit test (model correctness, on-TPU)

```bash
cd ~/tpu-inference
~/vllm-env/bin/python -m pytest tests/models/jax/test_granite.py -v
```

Expected: **3/3 PASS** (~3 min on v6e). The test uses a small mock
`ibm-granite/granite-3.3-2b-instruct` config and asserts:
- multipliers wired (`embedding_multiplier`, `residual_multiplier`,
  `attention_multiplier`),
- weight/output shapes and tied `lm_head`,
- forward pass with `create_kv_caches` (bf16 and fp8 KV cases),
- `logits_scaling` math.

## 3. Serve

```bash
# TP_SIZE [MODEL_LEN]. Defaults: full model len, --max-num-seqs 8, gpu-mem-util 0.90.
setsid bash -c "nohup ~/serve_granite.sh 2 < /dev/null > /dev/null 2>&1 &"   # TP=2, full 131072 ctx
# or: ~/serve_granite.sh 4 8192   # TP=4 capped at 8192
```

`serve_granite.sh` sets `SKIP_JAX_PRECOMPILE=1`, `HF_HUB_ENABLE_HF_TRANSFER=0`,
flags `--tensor-parallel-size`, `--gpu-memory-utilization 0.90`,
`--max-num-seqs 8`, `--download-dir /tmp/hf_cache`, optional `--max-model-len`,
and tees to `~/vllm_serve_granite.log`.

Launch detached (`setsid ... nohup`) — plain `nohup` over SSH loses the process
when the connection drops. Verify via log contents, not `pgrep -f` (self-match
gives false RUNNING).

**TP / context-length fit (bf16, fp8_e5m2 KV auto-selected on v6e):**

| TP | max ctx | KV cache tokens | Notes |
|----|---------|-----------------|-------|
| 1  | 84512 (auto-capped) | — | full 131072 OOMs (needs 20 GiB KV vs 12.9 GiB) |
| 2  | 131072 | 268,784 | full length fits |
| 4  | 8192 (example) | 637,184 | 2489 blocks |

## 4. Numerical validation vs HuggingFace

Compare vLLM (bf16, TPU) logits against HF (fp32, CPU) on fixed prompts.

```bash
# 1) On a CPU machine with the HF model: dump reference logits.
~/vllm-env/bin/python ~/hf_granite_logits.py        # -> /tmp/hf_granite_logits.npz

# 2) On the TPU host (server running): query vLLM logprobs and compare.
~/vllm-env/bin/python ~/cmp_granite_logits.py
```

Prompts: `"The capital of France is"`, `"17 * 24 ="`, `"def fibonacci(n):"`,
`"Once upon a time, in a small town,"`.

Expected (bf16 tolerance): top-1 identical on all prompts; top-5 overlap 5/5;
top-20 overlap ~19-20; max Δlogprob per prompt ≈ 0.036-0.093. This confirms the
Granite multipliers are numerically correct.

## 5. Benchmark

```bash
# vllm bench serve; NOTE --random-prefix-len 1 is required (dataset validation
# fails with DEFAULT_PREFIX_LEN=0). ignore-eos to force full output length.
~/run_bench.sh
```

`run_bench.sh` runs:
```
vllm bench serve --backend vllm --dataset-name random \
  --random-input-len 8192 --random-output-len 1024 --random-prefix-len 1 \
  --num-prompts 160 --max-concurrency 32 --ignore-eos \
  --save-result  # -> /tmp/bench_8k_1k_c32_p160.json ; log ~/bench_8k_1k.log
```

Reference result (TP=2, max-num-seqs 8, 8k-in/1k-out, concurrency 32, 160
prompts): 160/0 ok, ~451 tok/s output (peak 616), ~4061 tok/s total, TPOT mean
~16 ms, TTFT mean ~51 s. Engine logs show `Running: 8, Waiting: 8` →
**`--max-num-seqs 8` is the TTFT bottleneck**; raise it (and/or TP) for higher
throughput.

## 6. Troubleshooting

- **SSH exit 255 / dropped server:** always launch with `setsid bash -c
  "nohup ... < /dev/null &"`.
- **`vllm bench serve` "input+output > max_model_len" (HTTP 400):** lower
  `--random-input-len`/`--random-output-len` or raise `--max-model-len`.
- **`--random-input-len is too small`:** pass `--random-prefix-len 1`.
- **rsync to host fails** (hostname not in ssh config): use
  `tar czf /tmp/x.tgz ... && gcloud compute tpus tpu-vm scp` then extract on host.
- **vocab divisibility error sharding embed/lm_head (TP>1):** this branch's
  `vocab_pad_map` zero-pads vocab 49159→49160; ensure the model is loaded via
  this branch.
