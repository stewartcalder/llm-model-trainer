FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# NOTE: RunPod's GitHub build uses the REPO ROOT as the build context, so COPY
# paths are relative to the repo root (runpod_worker/…), not to this directory.

# Install Python deps
COPY runpod_worker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy handler
COPY runpod_worker/handler.py .

# RunPod expects the handler to be the CMD
CMD ["python", "-u", "handler.py"]
