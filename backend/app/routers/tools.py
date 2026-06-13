"""Tools endpoints. Currently hosts the screen-text-scraper tool."""
from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .. import models, schemas
from ..database import get_session
from ..screen_scraper import (
    availability,
    grab_full_screen_png,
    request_cancel,
    run_scrape_job,
)
from ..serialize import scrape_job_out

router = APIRouter(prefix="/api/tools/screen-scraper", tags=["tools"])


@router.get("/status", response_model=schemas.ScreenScraperStatusOut)
async def status():
    return await asyncio.to_thread(availability)


@router.get("/screenshot")
async def screenshot():
    """Return a PNG of the full primary monitor for the region/click picker."""
    try:
        png = await asyncio.to_thread(grab_full_screen_png)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(503, f"Could not capture the screen: {exc}")
    return Response(content=png, media_type="image/png",
                    headers={"Cache-Control": "no-store"})


@router.post("/start", response_model=schemas.ScrapeJobOut)
async def start(payload: schemas.ScrapeStartRequest, db: AsyncSession = Depends(get_session)):
    project = await db.get(models.Project, payload.project_id)
    if not project:
        raise HTTPException(404, "Project not found")

    avail = await asyncio.to_thread(availability)
    if not avail["available"]:
        raise HTTPException(503, avail["detail"] or "Screen scraper is not available.")

    config = payload.model_dump()
    job = models.ScrapeJob(
        project_id=payload.project_id,
        title=payload.title.strip(),
        status="queued",
        config_json=json.dumps(config),
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)

    asyncio.create_task(run_scrape_job(job.id, payload.project_id, config))
    return scrape_job_out(job)


@router.get("/jobs", response_model=list[schemas.ScrapeJobOut])
async def list_jobs(project_id: str, db: AsyncSession = Depends(get_session)):
    result = await db.execute(
        select(models.ScrapeJob)
        .where(models.ScrapeJob.project_id == project_id)
        .order_by(models.ScrapeJob.created_at.desc())
        .limit(20)
    )
    # Omit the (potentially large) accumulated text from list responses.
    return [scrape_job_out(j, include_text=False) for j in result.scalars().all()]


@router.get("/jobs/{job_id}", response_model=schemas.ScrapeJobOut)
async def get_job(job_id: str, db: AsyncSession = Depends(get_session)):
    job = await db.get(models.ScrapeJob, job_id)
    if not job:
        raise HTTPException(404, "Scrape job not found")
    return scrape_job_out(job)


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str, db: AsyncSession = Depends(get_session)):
    job = await db.get(models.ScrapeJob, job_id)
    if not job:
        raise HTTPException(404, "Scrape job not found")
    return {"cancelling": request_cancel(job_id)}
