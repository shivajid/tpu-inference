# Serving jina-embeddings-v2 (JinaBert) on TPU

Runbook for bringing up `jinaai/jina-embeddings-v2-small-en` — a JAX-native
encoder-only embedding model (symmetric ALiBi, GEGLU, post-LayerNorm) — on a
fresh TPU VM (validated on v6e, TP=1). The model runs under vLLM's pooling
runner with no KV cache; mean pooling executes in vLLM's CPU pooler.

## 1. Code

```bash
git clone https://github.com/shivajid/tpu-inference.git
cd tpu-inference
git checkout jina-v2-embeddings-clean
```

## 2. Python environment (uv, python 3.12)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.bashrc
uv venv ~/vllm-env --python 3.12 --seed    # --seed installs pip into the venv
source ~/vllm-env/bin/activate
```

## 3. Install vLLM TPU stack, then overlay the repo editable

```bash
python -m pip install vllm-tpu==0.26.0           # version this runbook was validated against
python -m pip install -e .                       # CRITICAL — see below
python -m pip install onnxruntime transformers   # for the parity test
```

(Validated environment: `vllm-tpu 0.26.0`, `jax/jaxlib 0.11.0`,
`libtpu 0.0.44`, python 3.12.)

The editable install is **required**: `vllm serve` spawns API-server and
engine subprocesses that import `tpu_inference` from site-packages, not from
your checkout. Without `-e .` those processes silently run the stale bundled
copy. Verify (must print the repo path, not site-packages):

```bash
cd ~ && python -c 'import tpu_inference, os; print(os.path.dirname(tpu_inference.__file__))'
cd ~/tpu-inference
```

## 4. Environment patches (one shot, idempotent)

```bash
python scripts/setup_jina_env.py
export HF_TOKEN=<your-token>   # optional; avoids HF rate limits
```

The script applies three patches to the active environment (see its
docstring for details): `transformers.onnx` and `transformers.pytorch_utils`
stubs for APIs removed in transformers v5 but still imported (never executed
at inference) by the Jina remote-code config, and a `sitecustomize.py` hook
that registers the JinaBert architectures with vLLM's ModelRegistry in every
process — needed because the API-server process validates ModelConfig before
`vllm.general_plugins` load or `tpu_inference` is imported.

## 5. Validate

```bash
pytest tests/models/jax/test_jina_bert.py -v -rs   # parity vs official ONNX export; expect 3 passed
pytest tests/e2e/test_jina_embeddings.py -v -rs    # full engine path
```

The parity test compares per-token hidden states and mean-pooled embeddings
against the model repo's official ONNX export (same weights, float32, no
remote code) and requires cosine similarity > 0.999.

## 6. Serve

```bash
vllm serve jinaai/jina-embeddings-v2-small-en --runner pooling --convert embed \
  --trust-remote-code --max-model-len 1024 --dtype float32
```

`--convert embed` is required: vLLM classifies the `JinaBertForMaskedLM`
architecture string as masked-LM and gates the embeddings API otherwise. No
pooler flag is needed — vLLM auto-detects mean pooling from the repo's
sentence-transformers configuration.

Query:

```bash
curl localhost:8000/v1/embeddings -H 'Content-Type: application/json' \
  -d '{"model": "jinaai/jina-embeddings-v2-small-en", "input": "How is the weather today?"}'
```

Expect a 512-dimensional embedding.

## Known limitations / follow-ups

- `max_model_len` capped at 1024–2048 for now: the dense ALiBi bias tensor is
  O(heads × T²); the full 8192 context wants kernel-side bias computation.
- float32 only (validated); bf16 pending validation against the fp32 baseline.
- TP=1 (model is ~33M params); slopes already shard with heads for TP > 1.
- The sitecustomize hook is a workaround; an upstream vLLM registration hook
  in the API-server process would remove it.
