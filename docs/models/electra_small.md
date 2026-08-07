# Serving ELECTRA-small (discriminator) on TPU

Runbook for bringing up `google/electra-small-discriminator` — a JAX-native
BERT-style encoder (learned absolute positions, factorized 128→256
embeddings, post-LayerNorm) — on a fresh TPU VM (targets v6e, TP=1). The
model runs under vLLM's pooling runner with no KV cache; pooling executes in
vLLM's CPU pooler (MEAN by default).

The replaced-token-detection head (`discriminator_predictions.*`) is dropped
at load time: this integration serves the pretrained encoder as an embedding
model. Note that raw ELECTRA embeddings are not contrastively tuned — for
semantic-search quality use a purpose-built embedder; this path is for
serving the ELECTRA encoder itself (feature extraction, downstream scoring).

## 1. Code

```bash
git clone https://github.com/shivajid/tpu-inference.git
cd tpu-inference
git checkout feat/electra-small-tpu-v6e
```

## 2. Python environment (uv, python 3.12)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.bashrc
uv venv ~/vllm-env --python 3.12 --seed
source ~/vllm-env/bin/activate
```

## 3. Install vLLM TPU stack, then overlay the repo editable

```bash
python -m pip install vllm-tpu==0.26.0        # version the jina runbook was validated against
python -m pip install -e .                    # CRITICAL — see below
python -m pip install transformers            # for the parity test
```

The editable install is **required**: `vllm serve` spawns API-server and
engine subprocesses that import `tpu_inference` from site-packages, not from
your checkout. Without `-e .` those processes silently run the stale bundled
copy. Verify (must print the repo path, not site-packages):

```bash
cd ~ && python -c 'import tpu_inference, os; print(os.path.dirname(tpu_inference.__file__))'
cd ~/tpu-inference
```

Unlike JinaBert, ELECTRA needs **no** remote-code or transformers shims
(`scripts/setup_jina_env.py` is not required): the config loads natively and
the architectures are registered with vLLM's ModelRegistry on
`tpu_inference` import (see `_register_vllm_compat_archs`). If the
API-server process still fails arch validation in your environment, add the
same `sitecustomize.py` hook the jina runbook describes, with
`ElectraForPreTraining`/`ElectraModel` pointing at
`tpu_inference.models.vllm.electra_compat:ElectraForPreTraining`.

## 4. Convert the checkpoint to safetensors (one shot)

The HF repo ships only `pytorch_model.bin`; tpu-inference's weight loader
reads `*.safetensors` exclusively. Convert once to a local directory:

```bash
export HF_TOKEN=<your-token>   # optional; avoids HF rate limits
python scripts/convert_electra_to_safetensors.py \
  --output ~/electra-small-discriminator-st
```

## 5. Validate

```bash
pytest tests/models/jax/test_electra.py -x -q     # HF parity (CPU ok)
```

## 6. Serve

```bash
vllm serve ~/electra-small-discriminator-st \
  --runner pooling \
  --dtype float32 \
  --max-model-len 512 \
  --no-enable-prefix-caching \
  --served-model-name google/electra-small-discriminator
```

`--max-model-len 512` matches `max_position_embeddings`; longer inputs would
index past the learned position table. Pooling defaults to MEAN (set by the
registry shim); use `--override-pooler-config '{"pooling_type": "CLS"}'` for
CLS pooling.

## 7. Query

```bash
curl -s http://localhost:8000/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model": "google/electra-small-discriminator",
       "input": ["How is the weather today?"]}' | python -m json.tool
```

## Implementation notes

- JAX model: `tpu_inference/models/jax/electra.py` (arch strings
  `ElectraForPreTraining`, `ElectraModel`).
- vLLM registry shim: `tpu_inference/models/vllm/electra_compat.py`
  (torch-only stub; never executed).
- Differences from JinaBert: learned absolute position embeddings (taken
  from `attention_metadata.input_positions`) instead of ALiBi; factorized
  embeddings with an `embeddings_project` 128→256 linear; standard BERT
  GELU MLP (`intermediate.dense`/`output.dense`) instead of GEGLU. The
  encoder attention is the plain `encoder_only_attention` path — the ALiBi
  kernel changes on this branch are inherited from the jina work and unused
  here.
