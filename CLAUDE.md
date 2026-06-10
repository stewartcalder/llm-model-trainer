# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the app

```bash
./run.sh          # build frontend + serve on http://localhost:8000
./run.sh dev      # backend :8000 + Vite dev server :5173 (hot reload)
```

The script auto-creates `backend/.venv` and installs deps on first run. The frontend build lands in `frontend/dist/` and is served as a SPA by FastAPI.

To run the backend directly (e.g. after code changes without restarting via `./run.sh`):

```bash
cd backend
.venv/bin/python -m uvicorn app.main:app --port 8000 --log-level warning
```

## Environment

All secrets live in `backend/.env` (git-ignored). Required keys:

```
DATABASE_URL=postgresql+asyncpg://user:pass@127.0.0.1:5433/lora_builder
ANTHROPIC_API_KEY=          # optional — leave blank, not empty string
OPENAI_API_KEY=             # optional — leave blank, not empty string
RUNPOD_API_KEY=             # optional, for cloud training
RUNPOD_ENDPOINT_ID=         # optional, for cloud training
```

**Empty string vs absent**: `os.environ.get("KEY") or "fallback"` is used throughout to handle keys that are present-but-empty in `.env`. Don't use the default-argument form `os.environ.get("KEY", "fallback")` for optional keys — it returns `""` when the key is explicitly set to nothing.

`OLLAMA_HOST` may be set by the Ollama process without a scheme or port (e.g. `0.0.0.0`). The training router normalises it: adds `http://`, resolves missing port to `:11434`, and replaces `0.0.0.0` with `localhost`.

## Database

PostgreSQL via asyncpg (not SQLite as the original spec described). Schema is auto-created by SQLAlchemy `create_all` on startup. There is no migration tool — schema changes require manual `ALTER TABLE` or dropping and recreating.

The training pipeline uses **psycopg2** (synchronous) for DB writes from the training thread because asyncpg cannot be used from a thread without its own event loop.

## Architecture

```
browser
  └─ React SPA (frontend/dist, served by FastAPI)
       └─ /api/*  ──▶  FastAPI (backend/app/main.py)
                          ├─ routers/projects, sources, samples, runs, exports
                          ├─ routers/training  ←─ local + RunPod training
                          ├─ pipeline.py       ←─ ingest→chunk→generate→validate
                          ├─ local_trainer.py  ←─ Unsloth training thread
                          └─ runpod_client.py  ←─ RunPod serverless REST/GraphQL
```

**Pipeline flow** (data collection): `Sources` tab → add PDFs/URLs → `Configure` tab (chunking, LLM, sample types) → `Run` tab (background asyncio task in `pipeline.py`) → `Review` tab → `Export` tab.

**Training flow**: `Training` tab → choose Local or RunPod provider → pick base model from Ollama table → configure LoRA params → start. Local jobs run via `asyncio.to_thread(run_local_training, ...)` to avoid blocking the event loop.

### Key non-obvious design decisions

**Local training thread model**: `run_local_training` in `local_trainer.py` is a blocking function run via `asyncio.to_thread`. It uses `psycopg2` (sync) for DB updates and a `threading.Event` in module-level `_cancel_flags` for cancellation. The HuggingFace `TrainerCallback.on_log` checks the event each training step.

**GGUF export — zero network traffic**: Both `save_pretrained_gguf` and Unsloth's `save_pretrained_merged(merged_16bit)` secretly download the full fp16 base model from HuggingFace, even when training used 4-bit. The correct approach is PEFT's own `model.merge_and_unload()` which dequantises each bnb-4bit layer already in VRAM, applies the LoRA delta, and returns a standard transformers model — no network access. After that, `merged_model.save_pretrained()` just writes the in-memory state dict. Then `convert_hf_to_gguf.py` (run via `sys.executable` so it finds the venv's torch) + `llama-quantize` handle the GGUF conversion step.

**llama.cpp first-use build**: `_ensure_llama_cpp()` patches `builtins.input` to auto-accept the system-package install prompt (Unsloth calls `input()` outside Colab/Kaggle). `cmake` must be installed on the host (`sudo apt-get install cmake`). The build takes ~10 min the first time; subsequent calls return immediately because `~/.unsloth/llama.cpp` already exists.

**Ollama model → HuggingFace mapping**: `_OLLAMA_HF_MAP` in `routers/training.py` maps Ollama tag names to Unsloth-optimised HF model IDs. `_ollama_to_hf()` tries exact match, then strips quantisation suffixes from the tag (e.g. `deepseek-r1:32b-qwen-distill-q4_K_M` → `deepseek-r1:32b`).

**Modelfile absolute path**: Unsloth writes a relative `FROM <filename>` in the Modelfile. Before calling `ollama create`, the code rewrites the `FROM` line to an absolute path so `ollama create` works regardless of working directory.

**RunPod data**: The full training dataset (approved samples) is base64-encoded and sent to RunPod in the job payload. Use local training for any proprietary data.

### Frontend

React + TypeScript, no UI library (plain CSS in `styles.css`). All shared CSS variables are in `:root`. `api.ts` is the single API client; `types.ts` holds all shared interfaces. Pages are in `src/pages/`; reusable UI primitives in `src/ui.tsx`.

In dev mode, Vite proxies `/api` to `http://localhost:8000`. In production, the built `dist/` is mounted as a static SPA by FastAPI — rebuild after frontend changes (`cd frontend && npm run build`).

Training page polling: active jobs are polled every 5 seconds via `setInterval` inside a `useEffect` that watches `activeJob?.id` and `activeJob?.status`. Polling stops when status leaves `["queued", "running", "IN_QUEUE", "IN_PROGRESS"]`.

## Local training dependencies

Install once, outside `./run.sh`:

```bash
# Python dev headers (required by Triton CUDA JIT)
sudo apt-get install -y python3.12-dev cmake

# Unsloth + training stack (quote version specs to avoid shell redirection)
cd backend
.venv/bin/pip install torch==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128
.venv/bin/pip install "torchvision>=0.26.0" --index-url https://download.pytorch.org/whl/cu128
.venv/bin/pip install unsloth "trl>=0.12" datasets accelerate bitsandbytes peft safetensors
.venv/bin/pip install psycopg2-binary
```

llama.cpp (`~/.unsloth/llama.cpp`) is built automatically on the first training run that reaches the GGUF export step.
