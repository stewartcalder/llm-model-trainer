# LoRA / QLoRA Training Data Builder

A single-user web app that turns curated source material (PDFs, web pages) into a
well-formed instruction-tuning dataset, then fine-tunes a LoRA adapter and pushes
the result directly to a local Ollama model server.

> **You curate. The app transforms and trains.** Add sources → configure → run → review → export → fine-tune → use.

---

## Quick start

```bash
./run.sh
```

Then open **http://localhost:8000**.

The first run creates a Python venv, installs backend deps, installs frontend deps,
builds the React app, and serves everything from one FastAPI process. It works
**fully offline** out of the box using the `mock` LLM provider — no API key needed.

### Development mode (hot reload)

```bash
./run.sh dev
```

Runs the FastAPI backend on `:8000` and the Vite dev server on `:5173` (open the
latter). API calls are proxied to the backend.

---

## Full workflow

```
Add Sources ──▶ Configure ──▶ Run ──▶ Review ──▶ Export   ──▶ Fine-Tune ──▶ Ollama
  PDF / URL     chunking,     ingest    approve /   JSONL /     Local GPU     ollama run
                sample types, chunk     reject /    RunPod      or RunPod     <your-model>
                LLM, budget   generate  edit        download    cloud GPU
                              validate
```

1. **Sources** — drop PDFs or paste URLs. Text is extracted with `pypdf` / `trafilatura`. Duplicate files are skipped (content-hash dedup).
2. **Configure** — sentence-window chunking (configurable window/overlap), pick sample types (Q&A / Instruction), choose an LLM provider/model, set concurrency and a USD budget cap.
3. **Run** — dry-run estimate first, then a background pipeline with a live progress bar, log stream, sample preview, running cost ticker, and cancel button. Each chunk × sample type makes a **generation** call and an optional **critic** call; clean samples auto-approve, flagged ones wait for review.
4. **Review** — filter/search the samples table, inline-edit any field, approve / reject per-row or in bulk.
5. **Export** — Alpaca / ShareGPT / OpenAI JSONL with a stratified train/val split and a reproducibility manifest.
6. **Fine-Tune** — train a LoRA adapter directly from the app:
   - **Local (Unsloth)** — runs on your GPU using Unsloth + TRL. Exports a GGUF and registers it in your local Ollama server automatically.
   - **RunPod (Cloud)** — submits the dataset to a RunPod serverless endpoint for training on cloud GPUs; returns the adapter files for download.

---

## LLM providers (data generation)

| Provider | Notes |
|---|---|
| `mock` | Deterministic, offline, no key. Default — use it to try the whole flow. |
| `anthropic` | Set `ANTHROPIC_API_KEY` in `backend/.env` or in Pipeline settings. |
| `ollama` | Any local OpenAI-compatible endpoint; default `http://localhost:11434/v1`. |

---

## Training providers

| Provider | Requirements | Output |
|---|---|---|
| **Local (Unsloth)** | NVIDIA GPU, `python3.12-dev`, `cmake`, Unsloth/torch installed (see CLAUDE.md) | GGUF quantised adapter merged with base model, registered in Ollama |
| **RunPod (Cloud)** | `RUNPOD_API_KEY` + `RUNPOD_ENDPOINT_ID` in `.env`, Docker image deployed | Adapter zip download. **Note: training data is sent to RunPod — don't use for proprietary data.** |

### Base model selection

The Training tab reads your local Ollama model list and maps each installed model to
its HuggingFace fine-tuning equivalent (e.g. `qwen3:14b` → `unsloth/Qwen3-14B`).
A **cached** badge appears when the HF weights are already in `~/.cache/huggingface/hub/`.

On first GGUF export, llama.cpp is cloned and built automatically (~10 min, one-time only).
Subsequent exports reuse the cached build.

---

## Environment setup

Create `backend/.env`:

```
DATABASE_URL=postgresql+asyncpg://user:pass@127.0.0.1:5433/lora_builder
ANTHROPIC_API_KEY=
OPENAI_API_KEY=
RUNPOD_API_KEY=
RUNPOD_ENDPOINT_ID=
```

Leave optional keys blank (not quoted empty strings).

---

## Project layout

```
backend/                 FastAPI + SQLAlchemy (async, PostgreSQL)
  app/
    main.py              app entry; serves API and the built SPA
    models.py            ORM: projects, sources, chunks, samples, runs, training_jobs
    pipeline.py          ingest → chunk → generate → validate (async, bounded concurrency)
    local_trainer.py     Unsloth training thread + llama.cpp GGUF export
    runpod_client.py     RunPod serverless REST + GraphQL client
    routers/training.py  Training endpoints; Ollama→HF model map; local + RunPod dispatch
    ingest.py            PDF / URL extraction
    chunking.py          sentence-window chunking (tiktoken)
    llm.py               provider abstraction (mock / anthropic / ollama)
    export.py            Alpaca / ShareGPT / OpenAI JSONL writers
    prompts.py           generation + critic prompt templates
    routers/             projects, sources, samples, runs, exports
frontend/                React + TypeScript + Vite SPA (no UI library)
  src/pages/             Dashboard, Sources, Configure, RunMonitor, Review, ExportPanel, Training
data/                    PostgreSQL-backed; uploads and exports on filesystem (git-ignored)
runpod_worker/           Docker-based RunPod serverless handler
app-documentation/       Product spec
```

---

## What's implemented

| Phase | Feature | Status |
|---|---|---|
| 1 | PDF + URL ingestion, sentence-window chunking | ✅ |
| 1 | Q&A + Instruction sample generation with critic | ✅ |
| 1 | Anthropic / Ollama / mock providers | ✅ |
| 1 | Alpaca / ShareGPT / OpenAI JSONL export | ✅ |
| 1 | Multi-project management | ✅ |
| 2 | PostgreSQL storage (upgraded from SQLite) | ✅ |
| 3 | Local LoRA training via Unsloth | ✅ |
| 3 | GGUF export + Ollama model registration | ✅ |
| 3 | RunPod serverless cloud training | ✅ |
| 3 | Ollama-installed model picker for base model selection | ✅ |
| — | Image sources, semantic chunking, HF Hub push, Parquet | ❌ Not yet |
