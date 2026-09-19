#!/usr/bin/env bash
# OpenAI-compatible Faster Qwen3-TTS API, configured for Tesla P100 / Pascal.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/.venv/bin/activate"

cuda_lib_dirs="$(find "$script_dir/.venv/lib" -path '*/site-packages/nvidia/*/lib' -type d -printf '%p:' 2>/dev/null)"
export LD_LIBRARY_PATH="${cuda_lib_dirs%:}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HF_HOME="/home/nexusjuan/Qwen3-TTS/.cache/huggingface"

exec python "$script_dir/examples/openai_server.py" \
  --host 127.0.0.1 --port 8000 \
  --backend ggml --quant Q8_0 \
  --qwentts-no-fa --qwentts-clamp-fp16 \
  --qwentts-ref-cache-dir "$script_dir/.qwentts_refs" \
  --model Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --ref-audio /home/nexusjuan/GPT-SoVITS/benchmark_samples/carl2_first9s.wav \
  --ref-text "What the hell is this here some some sort of gay out? What are you dancing about here? You are poor, is it not clear that I am going to just completely." \
  --language English
