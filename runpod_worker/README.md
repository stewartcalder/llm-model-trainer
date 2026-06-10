# RunPod Serverless Worker

This directory contains the serverless handler for LoRA / QLoRA fine-tuning on RunPod GPUs.

## How it works

1. The webapp base64-encodes the approved training samples and posts them to your RunPod serverless endpoint.
2. The worker decodes the dataset, loads the base model from HuggingFace, runs SFTTrainer with LoRA / QLoRA, and returns the adapter files base64-encoded in the job output.
3. The webapp receives the output, decodes the files, and saves them to `data/exports/models/<job_id>/` — then offers a zip download.

## Deployment steps

### 1. Build and push the Docker image

The Dockerfile's `COPY` paths are relative to the **repo root** (so RunPod's
GitHub build, which uses the repo root as context, works). Build from the repo
root and point `-f` at the Dockerfile:

```bash
# from the repository root
docker build -f runpod_worker/Dockerfile -t your-dockerhub-username/lora-trainer:latest .
docker push your-dockerhub-username/lora-trainer:latest
```

> **Using RunPod's GitHub integration instead?** You don't build locally at all —
> point the endpoint at this repo with Dockerfile path `runpod_worker/Dockerfile`
> and RunPod builds it (repo root is the context automatically).

### 2. Create a RunPod Serverless endpoint

1. Go to [runpod.io](https://www.runpod.io) → **Serverless** → **New Endpoint**.
2. Set **Container image** to `your-dockerhub-username/lora-trainer:latest`.
3. Choose a GPU type. Recommended minimums:
   - 7B models (4-bit): RTX 4090 (24 GB VRAM) or 1× A40
   - 13B models (4-bit): A100 40 GB or 2× RTX 4090
4. Set **Min workers** to `0` (scales to zero when idle — spin-up only billing).
5. Increase **Execution timeout** to at least `3600` seconds (1 h) — training takes time.
6. Click **Deploy**. Note the **Endpoint ID** shown in the dashboard.

### 3. Get your API key

Go to **Settings → API Keys** → **Create API key**. Copy it.

### 4. Add keys to backend/.env

```
RUNPOD_API_KEY=your_api_key_here
RUNPOD_ENDPOINT_ID=your_endpoint_id_here
```

Restart the backend: `cd backend && uvicorn app.main:app --reload`.

The Training tab in the webapp will now show the endpoint as **configured** and allow you to submit fine-tuning jobs.

## Notes on model access

- Some models (e.g. Llama-3) require a HuggingFace access token and accepting the license.
- Pass the token as a RunPod environment variable `HF_TOKEN` in your endpoint settings.
- The handler respects `trust_remote_code=True` for community models.
- For very large models, increase the endpoint's **Container disk** setting beyond the default 10 GB.

## Dataset format reference

| Format | JSON structure |
|---|---|
| `alpaca` | `{"instruction": "…", "input": "…", "output": "…"}` |
| `sharegpt` | `{"conversations": [{"from": "human", "value": "…"}, {"from": "gpt", "value": "…"}]}` |
| `openai` | `{"messages": [{"role": "user", "content": "…"}, {"role": "assistant", "content": "…"}]}` |
