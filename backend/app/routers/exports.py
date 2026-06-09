"""Export endpoints (spec 5.5 / 8.6)."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from .. import models, schemas
from ..database import get_session
from ..export import export_dataset
from ..serialize import load_json

router = APIRouter(prefix="/api/projects/{project_id}/export", tags=["export"])


@router.post("", response_model=schemas.ExportOut)
async def run_export(project_id: str, payload: schemas.ExportRequest,
                     db: AsyncSession = Depends(get_session)):
    project = await db.get(models.Project, project_id)
    if not project:
        raise HTTPException(404, "Project not found")

    result = await db.execute(
        select(models.Sample)
        .join(models.Chunk, models.Sample.chunk_id == models.Chunk.id)
        .join(models.Source, models.Chunk.source_id == models.Source.id)
        .where(models.Source.project_id == project_id)
    )
    samples = result.scalars().all()
    if not any(s.status in payload.include_statuses for s in samples):
        raise HTTPException(400, "No samples match the selected statuses to export.")

    source_count = await db.scalar(
        select(func.count(models.Source.id)).where(models.Source.project_id == project_id)
    ) or 0
    chunk_count = await db.scalar(
        select(func.count(models.Chunk.id))
        .join(models.Source).where(models.Source.project_id == project_id)
    ) or 0

    manifest_extra = {
        "config_snapshot": load_json(project.config_json, {}),
        "source_count": source_count,
        "chunk_count": chunk_count,
    }
    out = export_dataset(
        project, samples, payload.format, payload.train_split,
        payload.include_statuses, manifest_extra,
    )
    return schemas.ExportOut(**out)


@router.get("/download")
async def download(project_id: str, path: str):
    """Stream a previously generated export file. Guards against path escape."""
    from ..config import EXPORT_DIR

    target = Path(path).resolve()
    if not str(target).startswith(str(Path(EXPORT_DIR).resolve())):
        raise HTTPException(403, "Path outside export directory.")
    if not target.exists():
        raise HTTPException(404, "File not found.")
    return FileResponse(target, filename=target.name)
