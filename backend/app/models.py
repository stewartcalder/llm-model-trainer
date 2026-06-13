"""SQLAlchemy ORM models (spec section 7.3).

All timestamp columns use DateTime(timezone=True) so asyncpg receives
timezone-aware datetimes and PostgreSQL stores TIMESTAMPTZ.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    config_json: Mapped[str] = mapped_column(Text, default="{}")

    sources: Mapped[list["Source"]] = relationship(
        back_populates="project", cascade="all, delete-orphan"
    )
    runs: Mapped[list["Run"]] = relationship(
        back_populates="project", cascade="all, delete-orphan"
    )


class Source(Base):
    __tablename__ = "sources"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"))
    type: Mapped[str] = mapped_column(String)  # pdf | url | txt | docx | md
    path_or_url: Mapped[str] = mapped_column(Text)
    title: Mapped[str] = mapped_column(String, default="")
    status: Mapped[str] = mapped_column(String, default="pending")  # pending|processing|done|error
    content_hash: Mapped[str | None] = mapped_column(String, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    metadata_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    project: Mapped[Project] = relationship(back_populates="sources")
    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="source", cascade="all, delete-orphan"
    )


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    source_id: Mapped[str] = mapped_column(ForeignKey("sources.id", ondelete="CASCADE"))
    text: Mapped[str] = mapped_column(Text)
    token_count: Mapped[int] = mapped_column(Integer, default=0)
    page_or_section: Mapped[str | None] = mapped_column(String, nullable=True)
    char_start: Mapped[int] = mapped_column(Integer, default=0)
    char_end: Mapped[int] = mapped_column(Integer, default=0)

    source: Mapped[Source] = relationship(back_populates="chunks")
    samples: Mapped[list["Sample"]] = relationship(
        back_populates="chunk", cascade="all, delete-orphan"
    )


class Sample(Base):
    __tablename__ = "samples"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    chunk_id: Mapped[str] = mapped_column(ForeignKey("chunks.id", ondelete="CASCADE"))
    type: Mapped[str] = mapped_column(String)  # qa | instruction
    instruction: Mapped[str] = mapped_column(Text, default="")
    input: Mapped[str] = mapped_column(Text, default="")
    output: Mapped[str] = mapped_column(Text, default="")
    quality_json: Mapped[str] = mapped_column(Text, default="{}")
    # pending_review | approved | rejected | edited
    status: Mapped[str] = mapped_column(String, default="pending_review")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    edited_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    chunk: Mapped[Chunk] = relationship(back_populates="samples")


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"))
    config_snapshot_json: Mapped[str] = mapped_column(Text, default="{}")
    status: Mapped[str] = mapped_column(String, default="queued")  # queued|running|done|error|cancelled
    stage: Mapped[str] = mapped_column(String, default="queued")
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    chunks_processed: Mapped[int] = mapped_column(Integer, default=0)
    chunks_total: Mapped[int] = mapped_column(Integer, default=0)
    samples_generated: Mapped[int] = mapped_column(Integer, default=0)
    tokens_used: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    log: Mapped[str] = mapped_column(Text, default="")

    project: Mapped[Project] = relationship(back_populates="runs")


class ScrapeJob(Base):
    """A screen-text-scraper run: OCRs a screen region in a click-through loop.

    The accumulated OCR text is stored on the row and, on success, written out as
    a `Source` (type "screen") so it flows into the normal ingest -> chunk ->
    sample pipeline for LLM training.
    """

    __tablename__ = "scrape_jobs"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"))
    source_id: Mapped[str | None] = mapped_column(
        ForeignKey("sources.id", ondelete="SET NULL"), nullable=True
    )
    title: Mapped[str] = mapped_column(String, default="")
    # queued | running | done | error | cancelled
    status: Mapped[str] = mapped_column(String, default="queued")
    config_json: Mapped[str] = mapped_column(Text, default="{}")
    pages: Mapped[int] = mapped_column(Integer, default=0)
    text: Mapped[str] = mapped_column(Text, default="")
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    project: Mapped[Project] = relationship()


class TrainingJob(Base):
    """A LoRA fine-tuning job dispatched to a RunPod serverless endpoint."""

    __tablename__ = "training_jobs"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"))
    runpod_job_id: Mapped[str | None] = mapped_column(String, nullable=True)
    # queued | running | completed | failed | cancelled
    status: Mapped[str] = mapped_column(String, default="queued")
    config_json: Mapped[str] = mapped_column(Text, default="{}")
    log: Mapped[str] = mapped_column(Text, default="")
    model_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    project: Mapped[Project] = relationship()
