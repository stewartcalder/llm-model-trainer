"""End-to-end pipeline run: ingest -> chunk -> generate -> validate (spec 5.3).

Runs as an asyncio background task. Progress is persisted on the Run row so the
frontend can poll. Concurrency is bounded by a semaphore; an in-memory cancel
flag lets the UI stop a run.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone

from sqlalchemy import select

from . import models
from .chunking import sentence_window_chunks
from .config import price_for_model
from .database import SessionLocal
from .ingest import extract_file, extract_url
from .llm import complete_json, make_provider
from .prompts import DEFAULT_TEMPLATES

# Run id -> asyncio.Event used to request cancellation.
CANCEL_FLAGS: dict[str, asyncio.Event] = {}


def _now() -> datetime:
    return datetime.now(timezone.utc)


class RunContext:
    def __init__(self, run_id: str, config: dict):
        self.run_id = run_id
        self.config = config
        self.lock = asyncio.Lock()
        self.tokens_in = 0
        self.tokens_out = 0
        self.cost = 0.0
        self.samples = 0
        self.chunks_done = 0
        self.log_lines: list[str] = []
        self.cancel = CANCEL_FLAGS.setdefault(run_id, asyncio.Event())
        price = price_for_model(config.get("llm", {}).get("model", ""))
        self.price_in, self.price_out = price

    def log(self, msg: str) -> None:
        ts = _now().strftime("%H:%M:%S")
        self.log_lines.append(f"[{ts}] {msg}")
        # Keep the log bounded.
        if len(self.log_lines) > 500:
            self.log_lines = self.log_lines[-500:]

    def account(self, in_tok: int, out_tok: int) -> None:
        self.tokens_in += in_tok
        self.tokens_out += out_tok
        self.cost += (in_tok / 1_000_000) * self.price_in
        self.cost += (out_tok / 1_000_000) * self.price_out

    @property
    def tokens(self) -> int:
        return self.tokens_in + self.tokens_out


async def _persist_progress(ctx: RunContext, *, stage: str | None = None,
                            status: str | None = None, chunks_total: int | None = None) -> None:
    async with SessionLocal() as db:
        run = await db.get(models.Run, ctx.run_id)
        if not run:
            return
        if stage is not None:
            run.stage = stage
        if status is not None:
            run.status = status
        if chunks_total is not None:
            run.chunks_total = chunks_total
        run.chunks_processed = ctx.chunks_done
        run.samples_generated = ctx.samples
        run.tokens_used = ctx.tokens
        run.cost_usd = round(ctx.cost, 6)
        run.log = "\n".join(ctx.log_lines)
        await db.commit()


async def _ingest_sources(ctx: RunContext, project_id: str) -> None:
    """Process pending sources into chunks."""
    ctx.log("Stage: ingestion + chunking")
    await _persist_progress(ctx, stage="ingesting")
    chunk_cfg = ctx.config.get("chunking", {})
    window = int(chunk_cfg.get("window_tokens", 512))
    overlap = int(chunk_cfg.get("overlap_tokens", 64))

    async with SessionLocal() as db:
        result = await db.execute(
            select(models.Source).where(
                models.Source.project_id == project_id,
                models.Source.status.in_(["pending", "error"]),
            )
        )
        sources = result.scalars().all()

    for src in sources:
        if ctx.cancel.is_set():
            return
        async with SessionLocal() as db:
            db_src = await db.get(models.Source, src.id)
            db_src.status = "processing"
            await db.commit()
        try:
            # Extraction is blocking I/O — run in a thread.
            if src.type == "url":
                doc = await asyncio.to_thread(extract_url, src.path_or_url)
            else:
                doc = await asyncio.to_thread(extract_file, src.path_or_url)

            chunk_rows: list[models.Chunk] = []
            for label, text in doc.sections:
                for ch in sentence_window_chunks(text, window, overlap):
                    chunk_rows.append(
                        models.Chunk(
                            source_id=src.id,
                            text=ch.text,
                            token_count=ch.token_count,
                            page_or_section=label,
                            char_start=ch.char_start,
                            char_end=ch.char_end,
                        )
                    )

            async with SessionLocal() as db:
                db_src = await db.get(models.Source, src.id)
                db.add_all(chunk_rows)
                if doc.title and not db_src.title:
                    db_src.title = doc.title
                db_src.content_hash = doc.content_hash()
                db_src.status = "done"
                db_src.error = None
                await db.commit()
            ctx.log(f"Ingested '{src.title or src.path_or_url}' -> {len(chunk_rows)} chunks")
        except Exception as exc:  # noqa: BLE001
            async with SessionLocal() as db:
                db_src = await db.get(models.Source, src.id)
                db_src.status = "error"
                db_src.error = str(exc)
                await db.commit()
            ctx.log(f"ERROR ingesting '{src.path_or_url}': {exc}")


def _quality_flagged(quality: dict) -> bool:
    if quality.get("reject"):
        return True
    for dim in ("faithfulness", "completeness", "clarity"):
        if dim in quality and isinstance(quality[dim], (int, float)) and quality[dim] < 3:
            return True
    return False


async def _generate_for_chunk(ctx: RunContext, provider, sem: asyncio.Semaphore,
                              chunk: models.Chunk, sample_type: str,
                              templates: dict, use_critic: bool) -> None:
    async with sem:
        if ctx.cancel.is_set():
            return
        budget = float(ctx.config.get("budget_usd", 0) or 0)
        if budget and ctx.cost >= budget:
            return
        gen_tmpl = templates.get(f"{sample_type}_generation")
        if not gen_tmpl:
            return
        try:
            data, res = await complete_json(provider, gen_tmpl.format(chunk_text=chunk.text))
            ctx.account(res.input_tokens, res.output_tokens)
        except Exception as exc:  # noqa: BLE001
            ctx.log(f"ERROR generating {sample_type}: {exc}")
            return

        if sample_type == "qa":
            instruction = data.get("question", "").strip()
            output = data.get("answer", "").strip()
            input_text = ""
        else:  # instruction
            instruction = data.get("instruction", "").strip()
            output = data.get("output", "").strip()
            input_text = data.get("input", "").strip()
        if not instruction or not output:
            ctx.log(f"Skipped empty {sample_type} sample")
            return

        quality: dict = {}
        status = "pending_review"
        if use_critic:
            critic_tmpl = templates.get(f"{sample_type}_critic")
            if critic_tmpl:
                try:
                    fields = {
                        "chunk_text": chunk.text,
                        "question": instruction,
                        "answer": output,
                        "instruction": instruction,
                        "output": output,
                    }
                    quality, cres = await complete_json(provider, critic_tmpl.format(**fields))
                    ctx.account(cres.input_tokens, cres.output_tokens)
                    # Flagged samples wait for human review; clean ones auto-approve.
                    status = "pending_review" if _quality_flagged(quality) else "approved"
                except Exception as exc:  # noqa: BLE001
                    ctx.log(f"Critic failed for {sample_type}: {exc}")

        async with SessionLocal() as db:
            db.add(
                models.Sample(
                    chunk_id=chunk.id,
                    type=sample_type,
                    instruction=instruction,
                    input=input_text,
                    output=output,
                    quality_json=json.dumps(quality),
                    status=status,
                )
            )
            await db.commit()
        async with ctx.lock:
            ctx.samples += 1


async def _generate_samples(ctx: RunContext, project_id: str) -> None:
    ctx.log("Stage: sample generation")
    await _persist_progress(ctx, stage="generating")
    sample_types = ctx.config.get("sample_types", ["qa"])
    llm_cfg = ctx.config.get("llm", {})
    use_critic = bool(llm_cfg.get("use_critic", True))
    templates = {**DEFAULT_TEMPLATES, **(ctx.config.get("prompt_templates") or {})}
    provider = make_provider(llm_cfg)
    concurrency = max(1, int(ctx.config.get("concurrency", 5)))
    sem = asyncio.Semaphore(concurrency)

    # Only chunks that have no samples yet (lets runs be re-run incrementally).
    async with SessionLocal() as db:
        result = await db.execute(
            select(models.Chunk)
            .join(models.Source)
            .where(models.Source.project_id == project_id)
        )
        all_chunks = result.scalars().all()
        existing = await db.execute(select(models.Sample.chunk_id))
        have_samples = {row[0] for row in existing.all()}

    chunks = [c for c in all_chunks if c.id not in have_samples]
    total = len(chunks)
    await _persist_progress(ctx, chunks_total=total)
    ctx.log(f"{total} chunks to process x {len(sample_types)} sample type(s)")

    persist_every = 0
    for chunk in chunks:
        if ctx.cancel.is_set():
            ctx.log("Run cancelled.")
            return
        budget = float(ctx.config.get("budget_usd", 0) or 0)
        if budget and ctx.cost >= budget:
            ctx.log(f"Budget limit ${budget} reached — stopping generation.")
            break
        tasks = [
            _generate_for_chunk(ctx, provider, sem, chunk, st, templates, use_critic)
            for st in sample_types
        ]
        await asyncio.gather(*tasks)
        async with ctx.lock:
            ctx.chunks_done += 1
        persist_every += 1
        if persist_every >= 1:
            persist_every = 0
            await _persist_progress(ctx)


async def run_pipeline(run_id: str, project_id: str, config: dict) -> None:
    ctx = RunContext(run_id, config)
    try:
        await _persist_progress(ctx, status="running", stage="starting")
        ctx.log("Pipeline started.")
        await _ingest_sources(ctx, project_id)
        if ctx.cancel.is_set():
            await _finish(ctx, "cancelled")
            return
        await _generate_samples(ctx, project_id)
        await _finish(ctx, "cancelled" if ctx.cancel.is_set() else "done")
    except Exception as exc:  # noqa: BLE001
        ctx.log(f"FATAL: {exc}")
        await _finish(ctx, "error")
    finally:
        CANCEL_FLAGS.pop(run_id, None)


async def _finish(ctx: RunContext, status: str) -> None:
    ctx.log(f"Pipeline finished: {status}. Samples={ctx.samples} "
            f"tokens={ctx.tokens} cost=${round(ctx.cost, 4)}")
    await _persist_progress(ctx, stage=status, status=status)
    async with SessionLocal() as db:
        run = await db.get(models.Run, ctx.run_id)
        if run:
            run.finished_at = _now()
            await db.commit()


def request_cancel(run_id: str) -> bool:
    ev = CANCEL_FLAGS.get(run_id)
    if ev:
        ev.set()
        return True
    return False
