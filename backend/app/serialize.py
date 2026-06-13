"""Helpers to convert ORM rows into Pydantic-friendly dicts."""
from __future__ import annotations

import json
from typing import Any

from . import models, schemas


def load_json(text: str | None, default: Any) -> Any:
    if not text:
        return default
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return default


def project_out(p: models.Project, source_count: int = 0, sample_count: int = 0) -> schemas.ProjectOut:
    return schemas.ProjectOut(
        id=p.id, name=p.name, description=p.description or "",
        created_at=p.created_at, config=load_json(p.config_json, {}),
        source_count=source_count, sample_count=sample_count,
    )


def source_out(s: models.Source, chunk_count: int = 0, sample_count: int = 0) -> schemas.SourceOut:
    return schemas.SourceOut(
        id=s.id, type=s.type, path_or_url=s.path_or_url, title=s.title,
        status=s.status, error=s.error, chunk_count=chunk_count,
        sample_count=sample_count, created_at=s.created_at,
    )


def sample_out(s: models.Sample, source: models.Source, chunk_text: str = "") -> schemas.SampleOut:
    return schemas.SampleOut(
        id=s.id, chunk_id=s.chunk_id, source_id=source.id,
        source_title=source.title or source.path_or_url, type=s.type,
        instruction=s.instruction, input=s.input, output=s.output,
        quality=load_json(s.quality_json, {}), status=s.status,
        chunk_text=chunk_text, created_at=s.created_at, edited_at=s.edited_at,
    )


def scrape_job_out(j: models.ScrapeJob, include_text: bool = True) -> schemas.ScrapeJobOut:
    return schemas.ScrapeJobOut(
        id=j.id, project_id=j.project_id, source_id=j.source_id, title=j.title or "",
        status=j.status, config=load_json(j.config_json, {}), pages=j.pages,
        text=(j.text or "") if include_text else "", error=j.error,
        created_at=j.created_at, finished_at=j.finished_at,
    )


def run_out(r: models.Run) -> schemas.RunOut:
    return schemas.RunOut(
        id=r.id, status=r.status, stage=r.stage, started_at=r.started_at,
        finished_at=r.finished_at, chunks_processed=r.chunks_processed,
        chunks_total=r.chunks_total, samples_generated=r.samples_generated,
        tokens_used=r.tokens_used, cost_usd=r.cost_usd, log=r.log,
    )
