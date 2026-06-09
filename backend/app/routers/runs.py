"""Pipeline run control: dry-run estimate, start, monitor, cancel (spec 8.4)."""
from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from .. import models, schemas
from ..config import price_for_model
from ..database import get_session
from ..pipeline import request_cancel, run_pipeline
from ..serialize import load_json, run_out

router = APIRouter(prefix="/api/projects/{project_id}/runs", tags=["runs"])


@router.get("/dry-run", response_model=schemas.DryRunOut)
async def dry_run(project_id: str, db: AsyncSession = Depends(get_session)):
    project = await db.get(models.Project, project_id)
    if not project:
        raise HTTPException(404, "Project not found")
    config = load_json(project.config_json, {})

    pending_sources = await db.scalar(
        select(func.count(models.Source.id)).where(
            models.Source.project_id == project_id,
            models.Source.status.in_(["pending", "error"]),
        )
    ) or 0

    # Chunks already produced but not yet turned into samples.
    chunks_q = await db.execute(
        select(models.Chunk)
        .join(models.Source)
        .where(models.Source.project_id == project_id)
    )
    chunks = chunks_q.scalars().all()
    have = await db.execute(select(models.Sample.chunk_id))
    have_ids = {r[0] for r in have.all()}
    ready_chunks = [c for c in chunks if c.id not in have_ids]

    window = int(config.get("chunking", {}).get("window_tokens", 512))
    # Rough estimate: assume pending sources average ~6 chunks each.
    est_new_chunks = pending_sources * 6
    est_chunks = len(ready_chunks) + est_new_chunks

    sample_types = config.get("sample_types", ["qa"])
    use_critic = bool(config.get("llm", {}).get("use_critic", True))
    calls_per_chunk = len(sample_types) * (2 if use_critic else 1)
    est_calls = est_chunks * calls_per_chunk

    # Token estimate: prompt ~ window + 120 overhead, output ~ max_tokens/2.
    max_tokens = int(config.get("llm", {}).get("max_tokens", 1024))
    in_per_call = window + 120
    out_per_call = max_tokens // 2
    est_tokens = est_calls * (in_per_call + out_per_call)
    price_in, price_out = price_for_model(config.get("llm", {}).get("model", ""))
    est_cost = est_calls * (in_per_call / 1e6 * price_in + out_per_call / 1e6 * price_out)

    return schemas.DryRunOut(
        pending_sources=pending_sources,
        estimated_chunks=est_chunks,
        estimated_calls=est_calls,
        estimated_tokens=est_tokens,
        estimated_cost_usd=round(est_cost, 4),
    )


@router.post("/start", response_model=schemas.RunOut)
async def start_run(project_id: str, db: AsyncSession = Depends(get_session)):
    project = await db.get(models.Project, project_id)
    if not project:
        raise HTTPException(404, "Project not found")

    # Prevent overlapping runs.
    active = await db.scalar(
        select(models.Run).where(
            models.Run.project_id == project_id,
            models.Run.status.in_(["queued", "running"]),
        )
    )
    if active:
        raise HTTPException(409, "A run is already in progress.")

    config = load_json(project.config_json, {})
    run = models.Run(
        project_id=project_id,
        config_snapshot_json=json.dumps(config),
        status="queued", stage="queued",
    )
    db.add(run)
    await db.commit()
    await db.refresh(run)

    asyncio.create_task(run_pipeline(run.id, project_id, config))
    return run_out(run)


@router.get("", response_model=list[schemas.RunOut])
async def list_runs(project_id: str, db: AsyncSession = Depends(get_session)):
    result = await db.execute(
        select(models.Run)
        .where(models.Run.project_id == project_id)
        .order_by(models.Run.started_at.desc())
        .limit(20)
    )
    return [run_out(r) for r in result.scalars().all()]


@router.get("/{run_id}", response_model=schemas.RunOut)
async def get_run(project_id: str, run_id: str, db: AsyncSession = Depends(get_session)):
    run = await db.get(models.Run, run_id)
    if not run or run.project_id != project_id:
        raise HTTPException(404, "Run not found")
    return run_out(run)


@router.post("/{run_id}/cancel")
async def cancel_run(project_id: str, run_id: str, db: AsyncSession = Depends(get_session)):
    run = await db.get(models.Run, run_id)
    if not run or run.project_id != project_id:
        raise HTTPException(404, "Run not found")
    ok = request_cancel(run_id)
    return {"cancelling": ok}
