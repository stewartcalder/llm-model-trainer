# LoRA/QLoRA Training Data Builder — Product Specification

**Version:** 0.3  
**Status:** Phases 1–3 implemented  
**Author:** Stewart Simpson  

---

## 1. Overview

A desktop or web application that takes a curated set of raw source material (PDFs, images, web pages), uses LLM calls to synthesise instruction-tuning pairs from that material, and exports a well-formed training dataset ready to load into a fine-tuning pipeline (LoRA / QLoRA via Hugging Face `datasets`, Axolotl, LLaMA-Factory, or similar).

The user's job is curation and configuration. The app's job is transformation and formatting.

---

## 2. Problem Statement

Producing a quality fine-tuning dataset is the bottleneck before any LoRA run. The work is:

1. Chunking heterogeneous source material into meaningful segments
2. Writing instruction / input / output triples (or chat turns) that reflect the desired behaviour
3. Deduplicating, filtering, and validating the resulting rows
4. Exporting in the exact schema the trainer expects

This is currently done manually or with ad-hoc scripts. This app automates steps 1–4 while keeping the human in the loop for curation decisions.

---

## 3. Core Concepts

| Term | Meaning |
|---|---|
| **Source** | A PDF file, image file, or URL supplied by the user |
| **Chunk** | An atomic passage extracted from a source (typically 200–800 tokens) |
| **Sample** | A single training example: `{instruction, input, output}` or chat turns |
| **Dataset** | The validated collection of samples ready for export |
| **Format** | The target schema: Alpaca, ShareGPT, OpenAI JSONL, or custom |
| **Pipeline Run** | One end-to-end execution: ingest → chunk → generate → validate → export |

---

## 4. User Journey

```
1. Create Project
      │
      ▼
2. Add Sources  ──────────────────────────────────────┐
   (drag-drop PDFs / images / paste URLs)             │
      │                                               │
      ▼                                               │
3. Configure Pipeline                                 │
   - Chunking strategy                                │
   - Sample types to generate                         │
   - Target format & output path                      │
   - LLM provider / model                             │
      │                                               │
      ▼                                               │
4. Click Run                                          │
      │                                               │
      ▼                                               │
5. Monitor Progress (live log + sample preview)       │
      │                                               │
      ▼                                               │
6. Review & Curate                                    │
   - Approve / reject / edit individual samples       │
   - See quality metrics                              │
      │                                               │
      ▼                                               │
7. Export Dataset                                     │
   (JSONL / Parquet / HF datasets push)               │
      │                                               │
      └──────── iterate: add more sources ────────────┘
```

---

## 5. Functional Requirements

### 5.1 Source Ingestion

| # | Requirement |
|---|---|
| S1 | Accept PDF files (single or batch). Extract text per-page using `pdfplumber` or `pypdf`. For scanned PDFs, fall back to OCR via `pytesseract`. |
| S2 | Accept image files (PNG, JPEG, WEBP). Send to vision-capable LLM endpoint for content extraction and description. |
| S3 | Accept URLs. Fetch with `httpx`, parse with `trafilatura` or `readability-lxml` to extract clean article text. Handle JS-rendered pages via optional Playwright mode. |
| S4 | Display a source list with status indicators: pending / processing / done / error. |
| S5 | Allow per-source metadata: title, domain tag, language, quality weight. |
| S6 | Reject duplicate sources (hash-based dedup at ingest). |

### 5.2 Chunking

| # | Requirement |
|---|---|
| C1 | Sentence-window chunking: configurable window size (default 512 tokens) and overlap (default 64 tokens). |
| C2 | Semantic chunking: use embedding similarity to split at topic boundaries (optional, slower). |
| C3 | Document-level chunking: treat each PDF page or each article as one chunk (for short sources). |
| C4 | Preview chunked output before generating samples. User can manually split or merge chunks. |
| C5 | Chunks stored with provenance: source ID, page/section, character offsets. |

### 5.3 Sample Generation (LLM Pipeline)

This is the core transformation step. For each chunk, one or more LLM calls produce training samples.

#### 5.3.1 Sample Types

The user selects which types to generate per run:

| Type | Description | Use case |
|---|---|---|
| **Q&A** | Generate a question answerable from the chunk, with a comprehensive answer | Knowledge base, RAG distillation |
| **Instruction-following** | Generate a task instruction the chunk could be a response to | General instruction tuning |
| **Summarisation** | Instruction: "Summarise the following." Input: chunk. Output: LLM summary | Summarisation tasks |
| **Rewrite / Style** | Rewrite the chunk in a target style (formal, concise, domain-specific) | Style transfer |
| **Multi-turn chat** | Generate a 2–4 turn conversation grounded in the chunk | Chat fine-tuning |
| **Classification** | Generate a label + reasoning for the chunk (e.g. sentiment, topic) | Classifier training |
| **Extraction** | Generate structured JSON extracted from the chunk (entities, dates, etc.) | Information extraction |

Multiple types can be selected; the app runs one LLM call per type per chunk.

#### 5.3.2 Prompt Architecture

Each sample type has a **generation prompt template** and a **critic prompt template**.

**Generation prompt** (example for Q&A):

```
You are a training data author. Given the passage below, write one high-quality 
question and a thorough answer. The question must be answerable solely from the 
passage. Do not copy the passage verbatim. Output JSON only:
{"question": "...", "answer": "..."}

PASSAGE:
{chunk_text}
```

**Critic prompt** (self-critique, optional):

```
You are a dataset quality reviewer. Evaluate the following Q&A pair against the 
source passage. Score each dimension 1–5 and return JSON only:
{"faithfulness": N, "completeness": N, "clarity": N, "reject": bool, "reason": "..."}

SOURCE: {chunk_text}
Q: {question}
A: {answer}
```

Samples with `reject: true` or any dimension < 3 are flagged for human review rather than auto-included.

#### 5.3.3 LLM Provider Configuration

| Provider | Models supported |
|---|---|
| Anthropic API | claude-sonnet-4, claude-haiku-4 (fast/cheap for drafts) |
| OpenAI API | gpt-4o, gpt-4o-mini |
| Ollama (local) | Any locally served model (qwen3, llama3, mistral, etc.) |
| OpenAI-compatible | Any endpoint implementing `/v1/chat/completions` |

> **Implemented:** mock, anthropic, ollama. OpenAI API is supported via the openai-compatible path.

Provider, model, temperature, max_tokens, and API key are set in project settings. For image sources, the selected model must support vision.

#### 5.3.4 Concurrency & Rate Limiting

- Configurable parallelism: default 5 concurrent LLM calls
- Automatic retry with exponential backoff (3 attempts)
- Token budget display: estimated cost before run, running tally during run
- Hard stop if budget limit exceeded

### 5.4 Dataset Management

| # | Requirement |
|---|---|
| D1 | All generated samples stored in local SQLite database (one DB per project). |
| D2 | Each sample has: id, source_id, chunk_id, type, instruction, input, output, quality_scores JSON, status (pending_review / approved / rejected / edited). |
| D3 | Samples table supports full-text search. |
| D4 | Bulk actions: approve all, reject all flagged, filter by source / type / score. |
| D5 | Inline editing of any sample field. Edit logged with timestamp. |
| D6 | Dataset statistics panel: sample count by type, average quality scores, token distribution histogram, estimated dataset size. |

### 5.5 Export

| # | Requirement |
|---|---|
| E1 | Export only `approved` samples (default) or all non-rejected samples. |
| E2 | **Alpaca format** — `{"instruction": "...", "input": "...", "output": "..."}` JSONL |
| E3 | **ShareGPT format** — `{"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}` JSONL |
| E4 | **OpenAI fine-tune format** — `{"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}` JSONL |
| E5 | **Parquet** — columnar export compatible with `datasets.load_dataset("parquet", ...)` |
| E6 | **Hugging Face Hub push** — authenticate with HF token, push to a named dataset repo (public or private). |
| E7 | Train/validation split: configurable ratio (default 90/10), stratified by sample type. |
| E8 | Export manifest: JSON file listing source files, chunk counts, sample counts, export timestamp, config snapshot. |

---

## 6. Non-Functional Requirements

| Area | Requirement |
|---|---|
| **Performance** | A run of 50 PDFs (~200 chunks) should complete in under 10 minutes on a standard API connection at default parallelism. |
| **Offline-capable** | All processing (except LLM calls) works offline. With Ollama provider the entire pipeline is offline. |
| **Data residency** | Source files and generated data stored locally only unless HF Hub push is explicitly triggered. |
| **Reproducibility** | Each run stores its full config (prompt templates, model, temperature, chunking params) so results can be regenerated. |
| **Portability** | Project exported as a single folder: SQLite DB + source files + manifest. |

---

## 7. Application Architecture

### 7.1 Recommended Stack

Given your existing stack (Python/FastAPI, React/TypeScript, Tauri), the natural fit is:

```
┌─────────────────────────────────────────────────────┐
│                  Tauri Desktop Shell                │
│  ┌───────────────────┐   ┌────────────────────────┐ │
│  │  React/TypeScript  │   │   Rust (Tauri core)   │ │
│  │  Frontend UI       │◄──│   IPC / file system   │ │
│  └────────┬──────────┘   └────────────────────────┘ │
│           │ REST / WebSocket                         │
│  ┌────────▼──────────────────────────────────────┐  │
│  │          Python FastAPI Backend               │  │
│  │  ┌──────────────┐  ┌───────────────────────┐  │  │
│  │  │  Ingestion   │  │   LLM Pipeline        │  │  │
│  │  │  (pdf/img/   │  │   (async workers,     │  │  │
│  │  │   web)       │  │    prompt templates)  │  │  │
│  │  └──────────────┘  └───────────────────────┘  │  │
│  │  ┌──────────────────────────────────────────┐  │  │
│  │  │         SQLite (project DB)              │  │  │
│  │  └──────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

Alternatively, ship as a pure Python CLI + web UI (FastAPI + React SPA served from the same process) with no Tauri dependency, for easier distribution to non-developer users.

### 7.2 Key Python Dependencies

| Purpose | Library |
|---|---|
| PDF extraction | `pypdf` |
| Web fetch | `httpx`, `trafilatura` |
| Chunking / tokenisation | `tiktoken` |
| LLM calls | `anthropic`, `openai` (both support Ollama-compatible endpoints) |
| Database | `sqlalchemy` + `asyncpg` (PostgreSQL) |
| Local fine-tuning | `unsloth`, `trl`, `peft`, `bitsandbytes`, `torch` |
| GGUF conversion | `llama.cpp` (built to `~/.unsloth/llama.cpp` on first use) |
| Cloud training | RunPod serverless REST + GraphQL API |
| Task queue | `asyncio` with semaphore for concurrency; `threading.Event` for training cancellation |
| Sync DB writes from thread | `psycopg2-binary` |

### 7.3 Data Model (PostgreSQL)

```sql
projects        (id, name, description, created_at, config_json)
sources         (id, project_id, type, path_or_url, title, status, hash, metadata_json)
chunks          (id, source_id, text, token_count, page_or_section, char_start, char_end)
samples         (id, chunk_id, type, instruction, input, output,
                 quality_json, status, created_at, edited_at)
runs            (id, project_id, config_snapshot_json, status, stage, started_at, finished_at,
                 chunks_processed, chunks_total, samples_generated, tokens_used, cost_usd, log)
training_jobs   (id, project_id, runpod_job_id, status, config_json, log,
                 model_path, created_at, finished_at)
```

---

## 8. UI Screens

### 8.1 Project Dashboard
- Project name, dataset statistics (sources, chunks, samples, approved %)
- Recent run history with cost and sample counts
- Quick actions: Add Sources, Run Pipeline, Export

### 8.2 Sources Panel
- Sortable list: source name, type icon, status badge, chunk count, sample count
- Drag-and-drop upload zone
- URL input field
- Per-source actions: re-process, remove, view chunks

### 8.3 Pipeline Configuration
- Accordion panels: Chunking / Sample Types / LLM / Budget / Output Format
- Template editor for generation and critic prompts (with variable highlighting)
- Dry-run: show estimated chunk count, LLM call count, token estimate, cost

### 8.4 Run Monitor
- Progress bar: Ingestion → Chunking → Generating → Validating
- Live log stream (collapsible)
- Live sample preview: last 5 generated samples
- Running cost ticker
- Cancel button

### 8.5 Dataset Review
- Filterable table: source / type / status / quality score
- Row expand to show full sample fields + source chunk provenance
- Inline edit mode
- Approve / reject per-row or bulk
- Quality score histogram sidebar

### 8.6 Export Panel
- Format selector (Alpaca / ShareGPT / OpenAI / Parquet)
- Train/val split slider
- Sample count by status (approved, pending, rejected)
- Export to folder / Push to HF Hub
- Download manifest

---

## 9. Configuration File Schema

Projects persist their pipeline config as JSON alongside the SQLite DB:

```json
{
  "project_id": "uuid",
  "name": "My Domain Expert Dataset",
  "chunking": {
    "strategy": "sentence_window",
    "window_tokens": 512,
    "overlap_tokens": 64
  },
  "sample_types": ["qa", "instruction", "summarisation"],
  "llm": {
    "provider": "anthropic",
    "model": "claude-haiku-4-5-20251001",
    "temperature": 0.7,
    "max_tokens": 1024,
    "use_critic": true
  },
  "concurrency": 5,
  "budget_usd": 5.0,
  "export": {
    "format": "alpaca",
    "train_split": 0.9,
    "include_statuses": ["approved"]
  },
  "prompt_templates": {
    "qa_generation": "...",
    "qa_critic": "..."
  }
}
```

---

## 10. Implementation Status

### Phase 1 — MVP ✅ Complete

- PDF and URL source types
- Sentence-window chunking
- Q&A and Instruction sample types with critic
- Anthropic, Ollama, and mock providers
- Alpaca / ShareGPT / OpenAI JSONL export
- React web UI
- Multi-project management

### Phase 2 — Infrastructure ✅ Complete

- Upgraded storage to PostgreSQL (asyncpg)
- LLM provider status indicator in sidebar (live connectivity check)

### Phase 3 — Local & Cloud Training ✅ Complete

- **Local training (Unsloth)**: LoRA fine-tuning on local GPU, GGUF export, automatic Ollama model registration
- **RunPod cloud training**: serverless endpoint integration with job polling and adapter download
- **Ollama model picker**: maps installed Ollama models to HuggingFace training equivalents; shows HF cache status; handles quantised tag variants (e.g. `deepseek-r1:32b-qwen-distill-q4_K_M`)
- **One-time llama.cpp build**: auto-detected and built unattended on first GGUF export
- **Global HF cache**: fp16 base model download cached at `~/.cache/huggingface/hub/` across training runs

### Remaining / Phase 4

| Feature | Notes |
|---|---|
| Image source support | Vision model calls for content extraction |
| Semantic chunking | Embedding-based topic boundary detection |
| Multi-turn chat samples | Requires multi-call generation loop |
| HF Hub push | Needs `huggingface_hub` auth flow in UI |
| Tauri packaging | Wrap FastAPI + React into distributable desktop app |
| Prompt template library | Pre-built templates per domain (legal, medical, code) |
| Active learning loop | Surface lowest-confidence samples for human review first |
| Dataset versioning | Tag exports, diff between versions |

---

## 12. Open Questions

1. **Critic model**: Should the critic always use the same model as generation, or allow a cheaper/faster model for validation?
2. **Dedup at sample level**: Beyond source-level hash dedup, should near-duplicate samples (embedding similarity > 0.95) be auto-rejected?
3. **Multi-document samples**: Should the app support cross-chunk samples (e.g. a comparison question spanning two chunks from different sources)?
4. **Formatting of image output**: For image sources, should the generated `instruction` always reference that the input is an image description, or should the image content be treated as plain text?
5. **Distribution**: Ship as Python package (`pip install lora-data-builder`) or as packaged Tauri desktop app with embedded Python runtime?
