"""Dataset review & curation endpoints (spec 5.4 / 8.5)."""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from .. import models, schemas
from ..database import get_session
from ..serialize import load_json, sample_out

router = APIRouter(prefix="/api/projects/{project_id}/samples", tags=["samples"])


def _base_query(project_id: str):
    return (
        select(models.Sample, models.Source, models.Chunk)
        .join(models.Chunk, models.Sample.chunk_id == models.Chunk.id)
        .join(models.Source, models.Chunk.source_id == models.Source.id)
        .where(models.Source.project_id == project_id)
    )


@router.get("", response_model=list[schemas.SampleOut])
async def list_samples(
    project_id: str,
    status: str | None = None,
    type: str | None = None,
    source_id: str | None = None,
    search: str | None = None,
    limit: int = Query(200, le=1000),
    offset: int = 0,
    db: AsyncSession = Depends(get_session),
):
    q = _base_query(project_id)
    if status:
        q = q.where(models.Sample.status == status)
    if type:
        q = q.where(models.Sample.type == type)
    if source_id:
        q = q.where(models.Source.id == source_id)
    if search:
        like = f"%{search}%"
        q = q.where(or_(
            models.Sample.instruction.ilike(like),
            models.Sample.input.ilike(like),
            models.Sample.output.ilike(like),
        ))
    q = q.order_by(models.Sample.created_at.desc()).limit(limit).offset(offset)
    result = await db.execute(q)
    return [sample_out(s, src, chunk.text) for s, src, chunk in result.all()]


@router.patch("/{sample_id}", response_model=schemas.SampleOut)
async def update_sample(project_id: str, sample_id: str, payload: schemas.SampleUpdate,
                        db: AsyncSession = Depends(get_session)):
    row = await db.execute(_base_query(project_id).where(models.Sample.id == sample_id))
    found = row.first()
    if not found:
        raise HTTPException(404, "Sample not found")
    sample, source, chunk = found
    edited = False
    for field in ("instruction", "input", "output"):
        val = getattr(payload, field)
        if val is not None and val != getattr(sample, field):
            setattr(sample, field, val)
            edited = True
    if payload.status is not None:
        sample.status = payload.status
    if edited:
        sample.edited_at = datetime.now(timezone.utc)
        if payload.status is None:
            sample.status = "edited"
    await db.commit()
    await db.refresh(sample)
    return sample_out(sample, source, chunk.text)


@router.post("/bulk")
async def bulk_action(project_id: str, payload: schemas.BulkAction,
                      db: AsyncSession = Depends(get_session)):
    result = await db.execute(_base_query(project_id))
    rows = result.all()
    changed = 0
    ids = set(payload.ids or [])
    for sample, _src, _chunk in rows:
        if payload.action == "approve_all" and sample.status != "rejected":
            if sample.status != "approved":
                sample.status = "approved"; changed += 1
        elif payload.action == "reject_flagged":
            q = load_json(sample.quality_json, {})
            flagged = q.get("reject") or any(
                isinstance(q.get(d), (int, float)) and q[d] < 3
                for d in ("faithfulness", "completeness", "clarity")
            )
            if flagged and sample.status != "rejected":
                sample.status = "rejected"; changed += 1
        elif payload.action == "approve_ids" and sample.id in ids:
            sample.status = "approved"; changed += 1
        elif payload.action == "reject_ids" and sample.id in ids:
            sample.status = "rejected"; changed += 1
    await db.commit()
    return {"changed": changed}


@router.get("/stats", response_model=schemas.StatsOut)
async def stats(project_id: str, db: AsyncSession = Depends(get_session)):
    sources = await db.scalar(
        select(func.count(models.Source.id)).where(models.Source.project_id == project_id)
    ) or 0
    chunks = await db.scalar(
        select(func.count(models.Chunk.id))
        .join(models.Source).where(models.Source.project_id == project_id)
    ) or 0

    result = await db.execute(_base_query(project_id))
    rows = result.all()
    samples = [r[0] for r in rows]

    by_type: dict[str, int] = {}
    by_status: dict[str, int] = {}
    quality_acc: dict[str, list[float]] = {}
    buckets = {"<128": 0, "128-256": 0, "256-512": 0, "512-1024": 0, ">1024": 0}
    for s, _src, chunk in rows:
        by_type[s.type] = by_type.get(s.type, 0) + 1
        by_status[s.status] = by_status.get(s.status, 0) + 1
        q = load_json(s.quality_json, {})
        for dim in ("faithfulness", "completeness", "clarity"):
            if isinstance(q.get(dim), (int, float)):
                quality_acc.setdefault(dim, []).append(float(q[dim]))
        tc = chunk.token_count
        if tc < 128: buckets["<128"] += 1
        elif tc < 256: buckets["128-256"] += 1
        elif tc < 512: buckets["256-512"] += 1
        elif tc < 1024: buckets["512-1024"] += 1
        else: buckets[">1024"] += 1

    approved = by_status.get("approved", 0)
    approved_pct = round(100 * approved / len(samples), 1) if samples else 0.0
    avg_quality = {k: round(sum(v) / len(v), 2) for k, v in quality_acc.items() if v}
    histogram = [{"bucket": k, "count": v} for k, v in buckets.items()]

    return schemas.StatsOut(
        sources=sources, chunks=chunks, samples=len(samples), by_type=by_type,
        by_status=by_status, approved_pct=approved_pct, avg_quality=avg_quality,
        token_histogram=histogram,
    )
