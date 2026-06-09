"""Source ingestion endpoints: PDF upload, URL add, list, delete (spec 5.1 / 8.2)."""
from __future__ import annotations

import hashlib
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from .. import models, schemas
from ..config import UPLOAD_DIR
from ..database import get_session
from ..serialize import source_out

router = APIRouter(prefix="/api/projects/{project_id}/sources", tags=["sources"])


async def _counts(db: AsyncSession, source_id: str) -> tuple[int, int]:
    chunk_count = await db.scalar(
        select(func.count(models.Chunk.id)).where(models.Chunk.source_id == source_id)
    )
    sample_count = await db.scalar(
        select(func.count(models.Sample.id))
        .join(models.Chunk, models.Sample.chunk_id == models.Chunk.id)
        .where(models.Chunk.source_id == source_id)
    )
    return chunk_count or 0, sample_count or 0


@router.get("", response_model=list[schemas.SourceOut])
async def list_sources(project_id: str, db: AsyncSession = Depends(get_session)):
    result = await db.execute(
        select(models.Source)
        .where(models.Source.project_id == project_id)
        .order_by(models.Source.created_at)
    )
    out = []
    for s in result.scalars().all():
        cc, sc = await _counts(db, s.id)
        out.append(source_out(s, cc, sc))
    return out


@router.post("/url", response_model=schemas.SourceOut)
async def add_url(project_id: str, payload: schemas.UrlSourceCreate,
                  db: AsyncSession = Depends(get_session)):
    project = await db.get(models.Project, project_id)
    if not project:
        raise HTTPException(404, "Project not found")
    url = payload.url.strip()
    dup = await db.scalar(
        select(models.Source).where(
            models.Source.project_id == project_id,
            models.Source.path_or_url == url,
        )
    )
    if dup:
        raise HTTPException(409, "This URL is already a source in the project.")
    src = models.Source(
        project_id=project_id, type="url", path_or_url=url,
        title=(payload.title or "").strip(), status="pending",
    )
    db.add(src)
    await db.commit()
    await db.refresh(src)
    return source_out(src, 0, 0)


@router.post("/upload", response_model=list[schemas.SourceOut])
async def upload_pdfs(project_id: str, files: list[UploadFile] = File(...),
                      db: AsyncSession = Depends(get_session)):
    project = await db.get(models.Project, project_id)
    if not project:
        raise HTTPException(404, "Project not found")

    saved: list[schemas.SourceOut] = []
    for upload in files:
        data = await upload.read()
        if not data:
            continue
        digest = hashlib.sha256(data).hexdigest()
        # Hash-based dedup at ingest (spec S6).
        dup = await db.scalar(
            select(models.Source).where(
                models.Source.project_id == project_id,
                models.Source.content_hash == digest,
            )
        )
        if dup:
            continue
        dest = Path(UPLOAD_DIR) / f"{digest[:16]}-{Path(upload.filename).name}"
        dest.write_bytes(data)
        src = models.Source(
            project_id=project_id, type="pdf", path_or_url=str(dest),
            title=Path(upload.filename).stem, status="pending", content_hash=digest,
        )
        db.add(src)
        await db.commit()
        await db.refresh(src)
        saved.append(source_out(src, 0, 0))
    return saved


@router.delete("/{source_id}")
async def delete_source(project_id: str, source_id: str, db: AsyncSession = Depends(get_session)):
    src = await db.get(models.Source, source_id)
    if not src or src.project_id != project_id:
        raise HTTPException(404, "Source not found")
    # Remove uploaded file from disk for PDFs.
    if src.type == "pdf":
        try:
            Path(src.path_or_url).unlink(missing_ok=True)
        except OSError:
            pass
    await db.delete(src)
    await db.commit()
    return {"ok": True}


@router.get("/{source_id}/chunks", response_model=list[schemas.ChunkOut])
async def source_chunks(project_id: str, source_id: str, db: AsyncSession = Depends(get_session)):
    result = await db.execute(
        select(models.Chunk).where(models.Chunk.source_id == source_id)
    )
    out = []
    for c in result.scalars().all():
        sc = await db.scalar(
            select(func.count(models.Sample.id)).where(models.Sample.chunk_id == c.id)
        )
        out.append(schemas.ChunkOut(
            id=c.id, source_id=c.source_id, text=c.text, token_count=c.token_count,
            page_or_section=c.page_or_section, sample_count=sc or 0,
        ))
    return out
