"""Project CRUD and pipeline config. Single-user app keeps a default project."""
from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .. import models, schemas
from ..config import DEFAULT_CONFIG
from ..database import get_session
from ..prompts import DEFAULT_TEMPLATES
from ..serialize import load_json, project_out

router = APIRouter(prefix="/api/projects", tags=["projects"])


def _default_config() -> dict:
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))  # deep copy
    cfg["prompt_templates"] = dict(DEFAULT_TEMPLATES)
    return cfg


async def get_or_create_default(db: AsyncSession) -> models.Project:
    result = await db.execute(select(models.Project).order_by(models.Project.created_at))
    project = result.scalars().first()
    if project is None:
        project = models.Project(name="My Dataset", config_json=json.dumps(_default_config()))
        db.add(project)
        await db.commit()
        await db.refresh(project)
    return project


@router.get("", response_model=list[schemas.ProjectOut])
async def list_projects(db: AsyncSession = Depends(get_session)):
    await get_or_create_default(db)
    result = await db.execute(select(models.Project).order_by(models.Project.created_at))
    return [project_out(p) for p in result.scalars().all()]


@router.post("", response_model=schemas.ProjectOut)
async def create_project(payload: schemas.ProjectCreate, db: AsyncSession = Depends(get_session)):
    project = models.Project(name=payload.name, config_json=json.dumps(_default_config()))
    db.add(project)
    await db.commit()
    await db.refresh(project)
    return project_out(project)


@router.get("/{project_id}", response_model=schemas.ProjectOut)
async def get_project(project_id: str, db: AsyncSession = Depends(get_session)):
    project = await db.get(models.Project, project_id)
    if not project:
        raise HTTPException(404, "Project not found")
    return project_out(project)


@router.patch("/{project_id}", response_model=schemas.ProjectOut)
async def update_project(project_id: str, payload: schemas.ProjectUpdate,
                         db: AsyncSession = Depends(get_session)):
    project = await db.get(models.Project, project_id)
    if not project:
        raise HTTPException(404, "Project not found")
    if payload.name is not None:
        project.name = payload.name
    if payload.config is not None:
        merged = load_json(project.config_json, {})
        merged.update(payload.config)
        project.config_json = json.dumps(merged)
    await db.commit()
    await db.refresh(project)
    return project_out(project)
