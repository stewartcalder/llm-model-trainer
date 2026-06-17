# LoRA Model Trainer — Functional Specification

## 1. Purpose

LoRA Model Trainer is a self-hosted web application that takes an organisation's
own documents and turns them into a **fine-tuned, locally-runnable language
model**. It covers the full path end to end:

> raw documents → cleaned training examples → human-reviewed dataset →
> fine-tuned LoRA adapter → GGUF model registered in Ollama

The goal is to let a non-specialist produce a custom small language model that
has absorbed a specific body of knowledge (a product manual, a regulatory
corpus, an internal wiki, a scanned archive) without writing code, managing GPUs
by hand, or hand-authoring training data.

It runs as a single application on the user's own machine/server. Training can be
done either on the local GPU or burst out to rented cloud GPUs; the resulting
model always lands back on the local machine.

---

## 2. Who it is for

- **Domain owners** who have a corpus and want a model that "knows" it.
- **ML-curious practitioners** who want a guided pipeline rather than a stack of
  notebooks.
- **Privacy-sensitive teams** who need the option to keep proprietary data on
  local hardware (local training never sends data to a third party).

---

## 3. Core concepts (glossary)

| Term | Meaning |
|------|---------|
| **Project** | A self-contained workspace: its own sources, generated samples, runs, exports and training jobs. |
| **Source** | An input document: PDF, DOCX, TXT, Markdown, a web URL, or text captured by the screen scraper. |
| **Chunk** | A source split into overlapping windows of text sized for the LLM. |
| **Sample** | One training example (a Q&A pair or an instruction/response) generated from a chunk. |
| **Run** | One execution of the data pipeline (ingest → chunk → generate → validate). |
| **Critic** | An optional second LLM pass that scores each generated sample and auto-approves clean ones. |
| **Export** | A dataset written to disk as train/validation JSONL plus a manifest, in a chosen format. |
| **Training job** | A fine-tuning run (local or cloud) that produces a LoRA adapter and, optionally, a GGUF model in Ollama. |
| **Adapter (LoRA)** | The small set of trained weights that encode what the model learned, layered on top of a base model. |
| **GGUF** | The quantised model file format consumed by Ollama / llama.cpp. |

---

## 4. End-to-end user workflow

The UI is organised as a set of tabs that mirror the lifecycle:

1. **Projects / Dashboard** — create or pick a project; see counts and stats.
2. **Sources** — add the documents to learn from.
3. **Configure** — set chunking, choose the generation LLM, pick sample types,
   set budget/concurrency.
4. **Run** — execute the pipeline; watch live progress, token usage and cost.
5. **Review** — inspect, edit, approve or reject generated samples.
6. **Export** — write the approved dataset to disk in the desired format.
7. **Training** — fine-tune a base model on the dataset (local or cloud) and
   register the result in Ollama.
8. **Tools** — utilities; currently a screen-text scraper for digitising
   otherwise-unextractable content.

A user does not have to do every step in one sitting — all state is persisted per
project, and runs can be re-executed incrementally.

---

## 5. Functional areas

### 5.1 Projects

- Create, rename, describe and delete projects.
- Each project stores its own configuration (chunking, LLM settings, prompt
  templates, budget, etc.).
- Dashboard surfaces per-project statistics: source/chunk/sample counts, sample
  breakdown by type and by status, approved percentage, average quality scores,
  and a token-length histogram of the corpus.

### 5.2 Sources (ingestion)

- **Add sources** by file upload (PDF, DOCX, TXT, Markdown) or by URL.
- Sources from the **screen scraper** (see §5.8) appear here automatically.
- On a pipeline run, each pending/errored source is extracted to text:
  - PDFs are split per page; DOCX by heading/paragraph; web pages are fetched and
    cleaned to readable text.
  - A content hash and (where available) a document title are recorded.
- Per-source **status** is tracked (`pending → processing → done`/`error`) with
  the error message surfaced in the UI when extraction fails.

### 5.3 Configure (generation settings)

Per project, the user controls:

- **Chunking** — sentence-window size (default ~512 tokens) and overlap (default
  ~64 tokens).
- **LLM provider** for sample generation — `anthropic`, `ollama` (local models),
  or `mock` (offline/testing). API keys are read from server-side environment
  config; a connectivity probe reports provider/model/latency.
- **Sample types** to generate — **Q&A** and/or **Instruction-following**.
- **Critic on/off** — whether to run the quality-scoring second pass.
- **Concurrency** — how many chunks to process in parallel.
- **Budget** — an optional USD ceiling that halts generation when reached.
- **Prompt templates** — defaults are provided and can be overridden per project.

A **dry-run estimate** reports pending sources, expected chunk count, number of
LLM calls, estimated tokens and estimated cost before committing to a run.

### 5.4 Run (the pipeline)

- A run executes as a background task with live, pollable progress.
- Stages: **ingesting → generating → done** (or `cancelled`/`error`).
- For each chunk and each selected sample type, the generation LLM produces a
  candidate sample. If the **critic** is enabled, each sample is scored on
  dimensions such as faithfulness, completeness and clarity:
  - **Clean** samples are **auto-approved**.
  - **Flagged** samples (low score or explicit reject) are set to
    **pending review**.
- Runs are **incremental**: only chunks without samples yet are processed, so a
  run can be re-executed after adding sources without duplicating work.
- Live accounting of tokens in/out and running cost; a **budget cap** stops
  generation cleanly when hit.
- A run can be **cancelled** mid-flight from the UI.

### 5.5 Review

- Browse generated samples with their source, chunk text, type, status and
  quality scores.
- **Edit** any sample's instruction/input/output inline.
- **Approve / reject** individually or in bulk (e.g. approve all, reject all
  flagged, approve/reject a selected set).
- Status values: `pending_review`, `approved`, `rejected`, `edited`.

### 5.6 Export

- Export the dataset in one of three formats: **Alpaca**, **ShareGPT**, or
  **OpenAI** (chat) JSON.
- Choose which sample statuses to include (default: approved only) and the
  **train/validation split** ratio.
- Output is written to disk as a **train file**, a **validation file** and a
  **manifest** describing the export, with row counts reported back.

### 5.7 Training

Fine-tunes a base model on the project's approved dataset and (optionally) makes
the result immediately usable in **Ollama**. Two execution providers:

- **Local** — runs on the host GPU via Unsloth, in a background thread. Data
  never leaves the machine. A status panel reports whether the local training
  stack and a GPU are available.
- **RunPod (cloud)** — packages the dataset and config and submits a serverless
  job to a rented GPU. The worker trains the adapter, exports the GGUF on the GPU,
  and uploads it to a private Hugging Face repository. The backend then downloads
  the GGUF and registers it locally — so a cloud run lands in Ollama exactly like
  a local run.
  > Note: RunPod sends the approved dataset to a third-party GPU. Use **local**
  > training for proprietary data.

Configurable training parameters include: base model (chosen from the user's
installed Ollama models, mapped to the matching fine-tuning base), LoRA rank /
alpha / dropout, epochs, batch size, learning rate, max sequence length, 4-bit
(QLoRA) toggle, dataset format, GGUF quantisation level, and the target Ollama
model name.

Outputs:
- A **LoRA adapter** (always preserved as cheap insurance).
- A **GGUF model** registered in local **Ollama** under the chosen name, ready to
  `ollama run`.
- A downloadable adapter archive.

**Base-model selection** lists the user's installed Ollama models and maps each to
its best fine-tuning equivalent, indicating which are already cached locally.

**Reliability behaviour:**
- A **server-side reconciler** polls in-flight cloud jobs on a fixed interval,
  independent of any open browser tab, so a job that finishes while nobody is
  watching is still imported before its cloud output expires.
- If a cloud job reports completed but its output is no longer retrievable, the
  system **recovers the model deterministically** from the Hugging Face upload
  location rather than silently reporting a model-less success; if it genuinely
  cannot be retrieved, the job is marked **failed** with a clear message.
- An import interrupted by a backend restart is **resumed automatically** on
  next startup.
- The "Available in Ollama" confirmation is shown only when the model was
  actually registered, not merely when the job finished.

### 5.8 Tools — Screen-text scraper

For content that cannot be extracted normally (paginated viewer apps, scanned
document viewers, DRM'd readers), the scraper digitises the screen:

- The user captures a screenshot of the host's primary monitor, drags a rectangle
  over the region where text appears, and marks the point to click to advance to
  the next page.
- A background job loops: capture the region → OCR it → append the text → pause →
  click "next" → capture again. It stops when the page stops changing or a page
  limit is reached.
- The combined text is saved and **registered as a Source** so it flows into the
  normal pipeline.
- Screen capture, OCR and mouse control run on the **backend host**, not the
  browser. The Tools tab self-reports missing dependencies (OCR engine, capture
  libraries, a usable display) and disables itself when the host cannot support
  it. Coordinates are scaled correctly for HiDPI/scaled displays.

---

## 6. External integrations

| Integration | Role | Required? |
|-------------|------|-----------|
| **Ollama** (local) | Lists installable base models; receives the final trained model. | Needed for the model-selection UI and to run trained models. |
| **Anthropic API** | Optional LLM provider for sample generation/critic. | Optional |
| **OpenAI API** | Optional LLM provider. | Optional |
| **RunPod** (serverless) | Cloud GPU provider for fine-tuning. | Optional (only for cloud training) |
| **Hugging Face Hub** | Stores the cloud worker's adapter + GGUF for the backend to download. | Required for cloud training |
| **PostgreSQL** | Application database (projects, sources, chunks, samples, runs, jobs). | Required |

All secrets (DB URL, API keys, RunPod/HF tokens) live in server-side environment
configuration and are never exposed to the browser.

---

## 7. Data model (high level)

```
Project ─┬─ Source ── Chunk ── Sample
         ├─ Run                       (pipeline executions, with progress + cost)
         ├─ TrainingJob               (local or RunPod fine-tunes)
         └─ ScrapeJob                 (screen-scraper runs; may create a Source)
```

- **Project** owns everything; deleting it cascades.
- **Sample** carries the generated content, quality scores and review status.
- **Run** carries live progress (stage, chunks processed, samples, tokens, cost,
  log).
- **TrainingJob** carries provider, config, status, log, and the local model path.
- **ScrapeJob** carries capture config, accumulated OCR text and the resulting
  source link.

---

## 8. Non-functional characteristics

- **Single-deployment web app**: a browser SPA served by a Python backend; no
  separate services to operate beyond PostgreSQL (and optionally Ollama).
- **Background processing**: long operations (ingest/generate, training, scraping,
  cloud-job reconciliation) run asynchronously; the UI polls for progress.
- **Cost transparency**: generation runs estimate and then track token usage and
  USD cost, with an optional hard budget cap.
- **Incremental & resumable**: pipeline runs skip already-processed chunks; cloud
  imports survive backend restarts; cloud jobs are reconciled server-side.
- **Privacy posture**: local generation/training keep data on the host; cloud
  training is opt-in and clearly flagged as sending data off-host.
- **Resilience**: extraction, critic, and cleanup failures are caught and surfaced
  per item without aborting the whole run.

---

## 9. Explicitly out of scope / current limitations

- No multi-user accounts, roles or authentication — it is a single-operator tool.
- No schema migration tooling; database schema changes are applied manually.
- The screen scraper depends on host OS capabilities (display, OCR engine) and is
  disabled when unavailable.
- Cloud training requires a configured Hugging Face repository for the model to be
  retrievable; without it, only a small adapter (not an Ollama-ready model) can be
  returned.
- Model quality depends on corpus quality and human review; the tool assists but
  does not guarantee a good dataset.
