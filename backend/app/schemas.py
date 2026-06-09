"""Pydantic request/response schemas."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# ---- Projects ----
class ProjectCreate(BaseModel):
    name: str = Field(min_length=1)


class ProjectUpdate(BaseModel):
    name: str | None = None
    config: dict[str, Any] | None = None


class ProjectOut(BaseModel):
    id: str
    name: str
    created_at: datetime
    config: dict[str, Any]


# ---- Sources ----
class UrlSourceCreate(BaseModel):
    url: str = Field(min_length=1)
    title: str | None = None


class SourceOut(BaseModel):
    id: str
    type: str
    path_or_url: str
    title: str
    status: str
    error: str | None = None
    chunk_count: int = 0
    sample_count: int = 0
    created_at: datetime


# ---- Chunks ----
class ChunkOut(BaseModel):
    id: str
    source_id: str
    text: str
    token_count: int
    page_or_section: str | None = None
    sample_count: int = 0


# ---- Samples ----
class SampleOut(BaseModel):
    id: str
    chunk_id: str
    source_id: str
    source_title: str
    type: str
    instruction: str
    input: str
    output: str
    quality: dict[str, Any]
    status: str
    chunk_text: str = ""
    created_at: datetime
    edited_at: datetime | None = None


class SampleUpdate(BaseModel):
    instruction: str | None = None
    input: str | None = None
    output: str | None = None
    status: str | None = None


class BulkAction(BaseModel):
    action: str  # approve_all | reject_flagged | approve_ids | reject_ids
    ids: list[str] | None = None


# ---- Runs ----
class RunOut(BaseModel):
    id: str
    status: str
    stage: str
    started_at: datetime
    finished_at: datetime | None = None
    chunks_processed: int
    chunks_total: int
    samples_generated: int
    tokens_used: int
    cost_usd: float
    log: str


# ---- Stats / dry run ----
class DryRunOut(BaseModel):
    pending_sources: int
    estimated_chunks: int
    estimated_calls: int
    estimated_tokens: int
    estimated_cost_usd: float


class StatsOut(BaseModel):
    sources: int
    chunks: int
    samples: int
    by_type: dict[str, int]
    by_status: dict[str, int]
    approved_pct: float
    avg_quality: dict[str, float]
    token_histogram: list[dict[str, Any]]


# ---- Export ----
class ExportRequest(BaseModel):
    format: str = "alpaca"  # alpaca | sharegpt | openai
    train_split: float = 0.9
    include_statuses: list[str] = ["approved"]


class ExportOut(BaseModel):
    manifest: dict[str, Any]
    train_file: str
    val_file: str
    manifest_file: str
    train_count: int
    val_count: int
