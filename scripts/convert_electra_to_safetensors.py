#!/usr/bin/env python3
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
"""One-shot: export google/electra-small-discriminator to safetensors.

The HF repo for ELECTRA ships only `pytorch_model.bin`, but tpu-inference's
weight loader reads `*.safetensors` exclusively. This script downloads the
checkpoint once via transformers and re-saves it (weights + config +
tokenizer) into a local HF-format directory that `vllm serve` and the tests
can point at directly.

Usage:
    python scripts/convert_electra_to_safetensors.py \
        [--model google/electra-small-discriminator] \
        [--output ~/electra-small-discriminator-st]
"""

import argparse
import os


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",
                        default="google/electra-small-discriminator",
                        help="HF model id (or local dir) to convert.")
    parser.add_argument("--output",
                        default="~/electra-small-discriminator-st",
                        help="Output directory (HF format, safetensors).")
    args = parser.parse_args()

    from transformers import AutoModelForPreTraining, AutoTokenizer

    output = os.path.expanduser(args.output)
    print(f"Downloading {args.model} ...")
    model = AutoModelForPreTraining.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    print(f"Saving safetensors checkpoint to {output} ...")
    model.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)
    print("Done. Serve with:\n"
          f"  vllm serve {output} --runner pooling --dtype float32 "
          "--max-model-len 512 --no-enable-prefix-caching")


if __name__ == "__main__":
    main()
