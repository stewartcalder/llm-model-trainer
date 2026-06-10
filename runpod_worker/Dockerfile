FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# NOTE: RunPod's GitHub build uses the REPO ROOT as the build context, so COPY
# paths are relative to the repo root (runpod_worker/…), not to this directory.

# Install Python deps
COPY runpod_worker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── llama.cpp: convert_hf_to_gguf.py + llama-quantize for on-GPU GGUF export ──
# Built into the image so the worker can merge→GGUF→quantise without any
# network fetch at runtime. Only the quantize target is compiled (fast).
# gguf/sentencepiece/protobuf are what convert_hf_to_gguf.py needs at runtime;
# installed explicitly so we don't disturb the pinned transformers/trl stack.
RUN apt-get update && apt-get install -y --no-install-recommends git cmake build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && git clone --depth 1 https://github.com/ggerganov/llama.cpp /opt/llama.cpp \
    && cmake -S /opt/llama.cpp -B /opt/llama.cpp/build -DLLAMA_CURL=OFF -DGGML_NATIVE=OFF \
    && cmake --build /opt/llama.cpp/build --config Release -j --target llama-quantize \
    && pip install --no-cache-dir gguf sentencepiece protobuf
ENV LLAMA_CPP_DIR=/opt/llama.cpp

# Copy handler
COPY runpod_worker/handler.py .

# RunPod expects the handler to be the CMD
CMD ["python", "-u", "handler.py"]
