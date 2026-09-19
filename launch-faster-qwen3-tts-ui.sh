#!/usr/bin/env bash
# Faster Qwen3-TTS UI: P100 / Pascal-safe GGML Q8_0 configuration.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/.venv/bin/activate"

# Torch's bundled CUDA libraries are also needed by the optional transcription
# helper and keep this launch self-contained.
cuda_lib_dirs="$(find "$script_dir/.venv/lib" -path '*/site-packages/nvidia/*/lib' -type d -printf '%p:' 2>/dev/null)"
export LD_LIBRARY_PATH="${cuda_lib_dirs%:}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HF_HOME="/home/nexusjuan/Qwen3-TTS/.cache/huggingface"

# Restrict the UI to the tested model/backend so it cannot accidentally select
# a BF16 or larger-model configuration that is unsuitable for the P100.
export ACTIVE_MODELS="Qwen/Qwen3-TTS-12Hz-0.6B-Base"
export DEMO_DEFAULT_BACKEND="ggml"
export DEMO_AVAILABLE_BACKENDS="ggml"
export DEMO_GGML_QUANT="Q8_0"
export DEMO_QWENTTS_REF_CACHE_DIR="$script_dir/.qwentts_refs"
export DEMO_QWENTTS_USE_FA="0"
export DEMO_QWENTTS_CLAMP_FP16="1"

exec python "$script_dir/demo/server.py" \
  --host 127.0.0.1 --port 7863 \
  --model Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --backend ggml --quant Q8_0
