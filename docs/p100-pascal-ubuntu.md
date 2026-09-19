# Ubuntu / Tesla P100 (Pascal SM60)

This is the tested installation path for a Tesla P100 16 GB on Ubuntu with an
NVIDIA driver new enough for CUDA 12.1 runtime libraries. It uses the 0.6B Base
model, the qwentts.cpp GGML backend, Q8_0 weights, and CUDA streaming.

The ordinary `qwentts-cpp-python` CUDA wheels deliberately target newer GPUs
(SM75+). On a P100 they load but fail at first inference with `no kernel image
is available for execution on the device`. Build the native library for SM60 as
shown below.

## Prerequisites

```bash
sudo apt update
sudo apt install -y git cmake build-essential g++-10 cuda-toolkit-11-5
```

`nvcc 11.5` must use GCC 10; newer GCC headers cause CUDA compilation errors.
Check both before continuing:

```bash
nvcc --version
g++-10 --version
nvidia-smi
```

## Install Faster Qwen3-TTS

```bash
git clone https://github.com/nexusjuan12/faster-qwen3-tts.git
cd faster-qwen3-tts
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# P100-compatible Torch build. Do not install a newer Torch build: recent
# wheels have dropped Pascal (SM60) support.
pip install torch==2.5.1+cu121 torchaudio==2.5.1+cu121 \
  --index-url https://download.pytorch.org/whl/cu121
pip install -e . --no-deps
pip install qwen-tts-hf==0.1.1.post1 transformers==5.17.0 \
  soundfile numpy fastapi 'uvicorn[standard]' python-multipart \
  'huggingface-hub[oauth]>=1.5.0,<2.0' nano-parakeet
```

The CUDA libraries supplied by the Torch wheels must be discoverable at
runtime. The included UI launcher does this automatically. For a shell session:

```bash
export LD_LIBRARY_PATH="$(find "$PWD/.venv/lib" -path '*/site-packages/nvidia/*/lib' \
  -type d -printf '%p:' | sed 's/:$//')${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

## Build qwentts.cpp for SM60

Clone the wrapper beside this checkout, build the version used by the project,
then install it in the same virtual environment:

```bash
cd ..
git clone https://github.com/andimarafioti/qwentts-cpp-python.git
cd qwentts-cpp-python
mkdir -p third_party
git clone --recursive https://github.com/ServeurpersoCom/qwentts.cpp third_party/qwentts.cpp
git -C third_party/qwentts.cpp fetch --depth=1 origin 7df559a8ca25f66fee02970514ebe5f01dee9055
git -C third_party/qwentts.cpp checkout --detach FETCH_HEAD
git -C third_party/qwentts.cpp submodule update --init --recursive

CC=/usr/bin/gcc-10 CXX=/usr/bin/g++-10 \
  ../faster-qwen3-tts/.venv/bin/python scripts/build_native.py \
  --backend cuda --cuda-compiler /usr/bin/nvcc --clean --jobs 6 \
  --cmake-arg='-DCMAKE_CUDA_ARCHITECTURES=60-real' \
  --cmake-arg='-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-10'

../faster-qwen3-tts/.venv/bin/pip install --no-deps -e .
```

## Run the local UI

```bash
cd ../faster-qwen3-tts
./launch-faster-qwen3-tts-ui.sh
```

Open <http://127.0.0.1:7863>. The launcher deliberately limits the UI to the
tested configuration:

- `Qwen/Qwen3-TTS-12Hz-0.6B-Base`
- GGML `Q8_0`
- Flash attention disabled
- FP16 clamping enabled
- local-only binding (`127.0.0.1`)

The first reference upload creates cached `.spk` and `.rvq` files in
`.qwentts_refs`; later requests reuse them.

## OpenAI-compatible TTS endpoint

Start the P100-configured API server with:

```bash
./launch-openai-tts-api.sh
```

It binds only to `127.0.0.1:8000` and exposes `POST /v1/audio/speech` plus
`GET /health`. The bundled voice is named `default`; unknown voice names fall
back to it. WAV and PCM are streamed, while MP3 is returned once encoding is
complete.

```bash
curl http://127.0.0.1:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"tts-1","input":"Hey guys, how do I sound?","voice":"default","response_format":"wav"}' \
  --output response.wav
```

## Known results and limitations

On the P100, Q8_0 generated 11.36 seconds of speech in 11.50 seconds, with a
time to first playable chunk of about 1.23 seconds. Results vary modestly with
sampling and generated duration.

`Q4_K_M` is not currently usable with this runtime on CUDA: it reaches an
unsupported `q6_K` `GET_ROWS` operation. Use `Q8_0`.

The standard PyTorch/CUDA-graph backend runs in FP32 on a P100, but takes about
20 seconds for an 11.5-second response and does not deliver playable streaming
audio until generation completes. The GGML configuration above is the preferred
P100 path.
