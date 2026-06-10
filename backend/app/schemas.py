"""Pydantic request/response schemas."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# ---- Projects ----
class ProjectCreate(BaseModel):
    name: str = Field(min_length=1)
    description: str = ""


class ProjectUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    config: dict[str, Any] | None = None


class ProjectOut(BaseModel):
    id: str
    name: str
    description: str
    created_at: datetime
    config: dict[str, Any]
    source_count: int = 0
    sample_count: int = 0


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


# ---- LLM status ----
class LLMStatusOut(BaseModel):
    ok: bool
    provider: str
    model: str
    latency_ms: int
    detail: str = ""


# ---- Training ----
class TrainingConfig(BaseModel):
    provider: str = "local"                      # "local" | "runpod"
    base_model: str = "unsloth/Llama-3.2-1B-Instruct"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    num_epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 2e-4
    max_seq_length: int = 2048
    use_4bit: bool = True
    dataset_format: str = "alpaca"               # alpaca | sharegpt | openai
    include_statuses: list[str] = ["approved"]
    # GGUF / Ollama fields (used by both local and RunPod providers — RunPod
    # now exports the GGUF on the GPU and the backend runs `ollama create`).
    gguf_quantization: str = "q4_k_m"           # q4_k_m | q5_k_m | q8_0 | f16
    ollama_model_name: str = ""                  # name to register in local Ollama


class LocalStatusOut(BaseModel):
    available: bool
    version: str | None = None
    gpu: bool = False
    detail: str = ""


class TrainingJobOut(BaseModel):
    id: str
    project_id: str
    runpod_job_id: str | None = None
    status: str
    config: dict[str, Any]
    log: str
    model_path: str | None = None
    created_at: datetime
    finished_at: datetime | None = None


class RunPodStatusOut(BaseModel):
    configured: bool
    endpoint_id: str
    health: dict[str, Any]
