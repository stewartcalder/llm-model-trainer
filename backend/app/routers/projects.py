"""Project CRUD — each project is an independent training dataset."""
from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from .. import models, schemas
from ..config import DEFAULT_CONFIG
from ..database import get_session
from ..prompts import DEFAULT_TEMPLATES
from ..serialize import load_json, project_out

router = APIRouter(prefix="/api/projects", tags=["projects"])


def _default_config() -> dict:
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    cfg["prompt_templates"] = dict(DEFAULT_TEMPLATES)
    return cfg


async def _counts(db: AsyncSession, project_id: str) -> tuple[int, int]:
    src_count = await db.scalar(
        select(func.count(models.Source.id)).where(models.Source.project_id == project_id)
    ) or 0
    sample_count = await db.scalar(
        select(func.count(models.Sample.id))
        .join(models.Chunk, models.Sample.chunk_id == models.Chunk.id)
        .join(models.Source, models.Chunk.source_id == models.Source.id)
        .where(models.Source.project_id == project_id)
    ) or 0
    return src_count, sample_count


@router.get("", response_model=list[schemas.ProjectOut])
async def list_projects(db: AsyncSession = Depends(get_session)):
    result = await db.execute(select(models.Project).order_by(models.Project.created_at))
    projects = result.scalars().all()
    out = []
    for p in projects:
        sc, smc = await _counts(db, p.id)
        out.append(project_out(p, sc, smc))
    return out


@router.post("", response_model=schemas.ProjectOut)
async def create_project(payload: schemas.ProjectCreate, db: AsyncSession = Depends(get_session)):
    project = models.Project(
        name=payload.name,
        description=payload.description,
        config_json=json.dumps(_default_config()),
    )
    db.add(project)
    await db.commit()
    await db.refresh(project)
    return project_out(project, 0, 0)


@router.get("/{project_id}", response_model=schemas.ProjectOut)
async def get_project(project_id: str, db: AsyncSession = Depends(get_session)):
    project = await db.get(models.Project, project_id)
    if not project:
        raise HTTPException(404, "Project not found")
    sc, smc = await _counts(db, project_id)
    return project_out(project, sc, smc)


@router.patch("/{project_id}", response_model=schemas.ProjectOut)
async def update_project(project_id: str, payload: schemas.ProjectUpdate,
                         db: AsyncSession = Depends(get_session)):
    project = await db.get(models.Project, project_id)
    if not project:
        raise HTTPException(404, "Project not found")
    if payload.name is not None:
        project.name = payload.name
    if payload.description is not None:
        project.description = payload.description
    if payload.config is not None:
        merged = load_json(project.config_json, {})
        merged.update(payload.config)
        project.config_json = json.dumps(merged)
    await db.commit()
    await db.refresh(project)
    sc, smc = await _counts(db, project_id)
    return project_out(project, sc, smc)


@router.delete("/{project_id}")
async def delete_project(project_id: str, db: AsyncSession = Depends(get_session)):
    project = await db.get(models.Project, project_id)
    if not project:
        raise HTTPException(404, "Project not found")
    await db.delete(project)
    await db.commit()
    return {"ok": True}
