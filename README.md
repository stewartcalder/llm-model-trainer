# LoRA / QLoRA Training Data Builder

A single-user web app that turns curated source material (PDFs, web pages) into a
well-formed instruction-tuning dataset, ready to load into a LoRA / QLoRA pipeline.

> **You curate. The app transforms.** Add sources → configure → run → review → export.

This is the **Phase 1 MVP** from [the spec](app-documentation/lora-training-data-builder-spec.md):
PDF + URL sources, sentence-window chunking, Q&A + Instruction sample types,
Anthropic / Ollama providers (plus an offline **mock** provider), Alpaca JSONL
export, SQLite storage.

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

## How it works

```
Add Sources ──▶ Configure ──▶ Run ──▶ Review ──▶ Export
  PDF / URL     chunking,     ingest    approve /   Alpaca /
                sample types, chunk     reject /    ShareGPT /
                LLM, budget   generate  edit        OpenAI JSONL
                              validate
```

1. **Sources** — drop PDFs or paste URLs. Text is extracted with `pypdf` /
   `trafilatura`. Duplicate files are skipped (content-hash dedup).
2. **Pipeline** — sentence-window chunking (configurable window/overlap), pick
   sample types, choose an LLM provider/model, set concurrency and a USD budget.
3. **Run** — a dry-run estimate first, then a background pipeline with a live
   progress bar, log stream, sample preview, running cost ticker, and cancel.
   For each chunk × sample type the app makes a **generation** call and an
   optional **critic** call; clean samples auto-approve, flagged ones wait for review.
4. **Review** — filter/search the samples table, inline-edit any field, approve /
   reject per-row or in bulk. Quality scores below 3 are flagged.
5. **Export** — Alpaca / ShareGPT / OpenAI JSONL with a stratified train/val split
   and a reproducibility manifest (config snapshot + counts).

---

## LLM providers

| Provider | Notes |
|---|---|
| `mock` | Deterministic, offline, no key. Default — use it to try the whole flow. |
| `anthropic` | Set the API key in Pipeline settings or the `ANTHROPIC_API_KEY` env var. |
| `ollama` | Any local OpenAI-compatible endpoint; set the base URL (default `http://localhost:11434/v1`). |

---

## Project layout

```
backend/                 FastAPI + SQLAlchemy (async) + the pipeline
  app/
    main.py              app entry; serves API and the built SPA
    models.py            ORM: projects, sources, chunks, samples, runs
    ingest.py            PDF / URL extraction
    chunking.py          sentence-window chunking (tiktoken)
    llm.py               provider abstraction (mock / anthropic / ollama)
    pipeline.py          ingest → chunk → generate → validate (async, bounded concurrency)
    export.py            Alpaca / ShareGPT / OpenAI JSONL writers
    prompts.py           generation + critic templates
    routers/             projects, sources, samples, runs, exports
frontend/                React + TypeScript + Vite SPA
  src/pages/             Dashboard, Sources, Configure, RunMonitor, Review, ExportPanel
data/                    SQLite DB, uploaded sources, exports (git-ignored)
```

All data stays local under `data/` unless you explicitly export.

---

## Not yet (Phase 2)

Image sources, semantic / multi-turn chat samples, HF Hub push, Parquet,
OCR for scanned PDFs, multi-project management, Tauri packaging. See the spec.
