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
"""One-shot environment setup for serving JinaBert (jina-embeddings-v2).

Applies three idempotent patches to the ACTIVE python environment. Run it
with the same interpreter that runs vLLM:

    python scripts/setup_jina_env.py

1. transformers.onnx stub — the jina-bert-implementation remote-code config
   imports `OnnxConfig` from a module removed in transformers v5.
2. transformers.pytorch_utils shims — `find_pruneable_heads_and_indices`,
   `prune_linear_layer`, `apply_chunking_to_forward` were removed in v5 but
   are imported (not executed at inference) by the remote modeling code.
3. sitecustomize.py registry hook — `vllm serve` spawns an API-server
   process that validates ModelConfig BEFORE loading vllm.general_plugins
   or importing tpu_inference, so in-repo registration cannot fire there.
   The hook registers the JinaBert archs the moment vLLM's registry module
   is imported, in every process.

Also remember: install tpu-inference editable (`pip install -e .`) so all
vLLM subprocesses import the repo code rather than a stale site-packages
snapshot.
"""

import os
import site
import sys
import textwrap

MARKER = "jina-bert v5-compat shim"


def patch_transformers_onnx():
    import transformers
    d = os.path.join(os.path.dirname(transformers.__file__), "onnx")
    init = os.path.join(d, "__init__.py")
    if os.path.isdir(d) and os.path.exists(init):
        print(f"[skip] transformers.onnx exists: {init}")
        return
    os.makedirs(d, exist_ok=True)
    with open(init, "w") as f:
        f.write(
            f"# {MARKER}: stub for removed transformers.onnx (only\n"
            "# referenced by jina-bert-implementation for ONNX export).\n"
            "class OnnxConfig:\n"
            "    @classmethod\n"
            "    def from_model_config(cls, *a, **k):\n"
            "        return cls()\n")
    print(f"[ok]   wrote {init}")


def patch_pytorch_utils():
    import transformers
    p = os.path.join(os.path.dirname(transformers.__file__),
                     "pytorch_utils.py")
    src = open(p).read()
    if "find_pruneable_heads_and_indices" in src:
        print(f"[skip] pytorch_utils already has shims: {p}")
        return
    src += textwrap.dedent(f'''

    # --- {MARKER}: APIs removed in transformers v5, imported (not executed
    # at inference) by pre-v5 remote code such as jina-bert-implementation ---
    import torch as _torch

    def find_pruneable_heads_and_indices(heads, n_heads, head_size,
                                         already_pruned_heads):
        mask = _torch.ones(n_heads, head_size)
        heads = set(heads) - already_pruned_heads
        for head in heads:
            head = head - sum(1 if h < head else 0
                              for h in already_pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = _torch.arange(len(mask))[mask].long()
        return heads, index

    def prune_linear_layer(layer, index, dim=0):
        raise NotImplementedError("pruning not supported (compat shim)")

    def apply_chunking_to_forward(forward_fn, chunk_size, chunk_dim,
                                  *input_tensors):
        return forward_fn(*input_tensors)
    ''')
    open(p, "w").write(src)
    print(f"[ok]   patched {p}")


SITECUSTOMIZE_HOOK = '''
# --- {marker}: register TPU JinaBert archs with vLLM's ModelRegistry in
# every python process (vllm serve validates ModelConfig in a spawned
# API-server process before any plugin or tpu_inference import). ---
import sys as _jina_sys
from importlib import abc as _jina_abc, util as _jina_util

_JINA_TARGET = "vllm.model_executor.models.registry"


class _JinaRegFinder(_jina_abc.MetaPathFinder):

    def find_spec(self, name, path=None, target=None):
        if name != _JINA_TARGET:
            return None
        _jina_sys.meta_path.remove(self)
        spec = _jina_util.find_spec(name)
        if spec is None or spec.loader is None:
            return None
        _orig = spec.loader.exec_module

        def exec_module(module, _orig=_orig):
            _orig(module)
            try:
                for arch in ("JinaBertForMaskedLM", "JinaBertModel"):
                    module.ModelRegistry.register_model(
                        arch,
                        "tpu_inference.models.vllm.jina_bert_compat:"
                        "JinaBertForMaskedLM")
            except Exception:
                pass

        spec.loader.exec_module = exec_module
        return spec


_jina_sys.meta_path.insert(0, _JinaRegFinder())
'''.format(marker=MARKER)


def patch_sitecustomize():
    sc = os.path.join(site.getsitepackages()[0], "sitecustomize.py")
    if os.path.exists(sc) and MARKER in open(sc).read():
        print(f"[skip] sitecustomize already hooked: {sc}")
        return
    with open(sc, "a" if os.path.exists(sc) else "w") as f:
        f.write(SITECUSTOMIZE_HOOK)
    print(f"[ok]   hooked {sc}")


def main():
    print(f"python: {sys.executable}")
    patch_transformers_onnx()
    patch_pytorch_utils()
    patch_sitecustomize()
    print("done. Reminder: install tpu-inference editable "
          "(`pip install -e .` from the repo root).")


if __name__ == "__main__":
    main()
