#!/bin/bash
# Copyright 2025 Google LLC
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

# Serve wrapper for Qwen/Qwen3.8-27B on TPU v6e.
# Sets the required env for the torchax path and starts the OpenAI-compatible
# vLLM server. All knobs can be overridden via environment variables.

set -e

# --- Required env for the Qwen3.5-hybrid torchax path (can be pre-set) ---
export MODEL_IMPL_TYPE="${MODEL_IMPL_TYPE:-vllm}"
export NEW_MODEL_DESIGN="${NEW_MODEL_DESIGN:-1}"
export SLICE_ROPE_CACHE="${SLICE_ROPE_CACHE:-1}"
export RAGGED_GATED_DELTA_RULE_IMPL="${RAGGED_GATED_DELTA_RULE_IMPL:-chunked_kernel_p_recurrent_kernel_d}"
export HF_HOME="${HF_HOME:-/hf_cache}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-/jax_cache}"

# --- Model / server knobs ---
MODEL="${MODEL:-Qwen/Qwen3.8-27B}"
TP="${TENSOR_PARALLEL_SIZE:-4}"
# 32k default is a practical production context. Only 16/64 layers are
# full-attention, so KV-cache growth with length is modest; the 48 GDN layers
# hold constant-size state. Model supports up to 262144 (raise MAX_MODEL_LEN
# and/or lower MAX_NUM_SEQS if you have the HBM headroom).
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
GPU_MEM_UTIL="${GPU_MEMORY_UTILIZATION:-0.9}"
PORT="${PORT:-8000}"

echo "Starting vLLM server: model=${MODEL} tp=${TP} max_model_len=${MAX_MODEL_LEN} " \
     "max_num_seqs=${MAX_NUM_SEQS} port=${PORT}"
echo "HF_HOME=${HF_HOME} JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR}"

exec vllm serve "${MODEL}" \
  --tensor-parallel-size "${TP}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --language-model-only \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --port "${PORT}"
