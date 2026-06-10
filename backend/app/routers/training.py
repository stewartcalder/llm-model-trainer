"""Training endpoints — supports local (Unsloth) and RunPod (serverless) providers."""
from __future__ import annotations

import asyncio
import base64
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .. import models, schemas
from ..config import EXPORT_DIR
from ..database import get_session
from ..runpod_client import cancel_job, health_check, job_status, list_gpu_types, submit_job
from ..serialize import load_json

router = APIRouter(prefix="/api/projects/{project_id}/training", tags=["training"])


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _job_out(j: models.TrainingJob) -> schemas.TrainingJobOut:
    return schemas.TrainingJobOut(
        id=j.id,
        project_id=j.project_id,
        runpod_job_id=j.runpod_job_id,
        status=j.status,
        config=load_json(j.config_json, {}),
        log=j.log,
        model_path=j.model_path,
        created_at=j.created_at,
        finished_at=j.finished_at,
    )


async def _build_dataset(project_id: str, db: AsyncSession,
                         cfg: schemas.TrainingConfig) -> tuple[str, int]:
    """Return (jsonl_text, sample_count)."""
    result = await db.execute(
        select(models.Sample)
        .join(models.Chunk, models.Sample.chunk_id == models.Chunk.id)
        .join(models.Source, models.Chunk.source_id == models.Source.id)
        .where(models.Source.project_id == project_id,
               models.Sample.status.in_(cfg.include_statuses))
    )
    samples = result.scalars().all()
    if not samples:
        raise HTTPException(400, "No samples match the selected statuses — approve some first.")

    from ..export import FORMATTERS
    formatter = FORMATTERS.get(cfg.dataset_format)
    if not formatter:
        raise HTTPException(400, f"Unknown dataset format: {cfg.dataset_format}")

    lines = [json.dumps(formatter(s), ensure_ascii=False) for s in samples]
    return "\n".join(lines) + "\n", len(samples)


# ── Ollama model → HuggingFace mapping ──────────────────────────────────────

# Maps the Ollama model tag (lowercased, :latest stripped) to the best
# Unsloth-optimised HuggingFace equivalent for fine-tuning.
_OLLAMA_HF_MAP: dict[str, str] = {
    # Llama
    "llama3.2:1b":         "unsloth/Llama-3.2-1B-Instruct",
    "llama3.2:3b":         "unsloth/Llama-3.2-3B-Instruct",
    "llama3.1:8b":         "unsloth/Llama-3.1-8B-Instruct",
    "llama3.1:70b":        "unsloth/Llama-3.1-70B-Instruct",
    "llama3:8b":           "unsloth/llama-3-8b-Instruct",
    # Mistral / Mixtral
    "mistral:7b":          "unsloth/mistral-7b-instruct-v0.3",
    "mistral":             "unsloth/mistral-7b-instruct-v0.3",
    "mixtral:8x7b":        "unsloth/Mixtral-8x7B-Instruct-v0.1",
    # Phi
    "phi3:mini":           "unsloth/Phi-3-mini-4k-instruct",
    "phi3.5:mini":         "unsloth/Phi-3.5-mini-instruct",
    "phi4:14b":            "unsloth/phi-4",
    "phi4":                "unsloth/phi-4",
    # Qwen 2.5
    "qwen2.5:0.5b":        "unsloth/Qwen2.5-0.5B-Instruct",
    "qwen2.5:1.5b":        "unsloth/Qwen2.5-1.5B-Instruct",
    "qwen2.5:3b":          "unsloth/Qwen2.5-3B-Instruct",
    "qwen2.5:7b":          "unsloth/Qwen2.5-7B-Instruct",
    "qwen2.5:14b":         "unsloth/Qwen2.5-14B-Instruct",
    "qwen2.5:32b":         "unsloth/Qwen2.5-32B-Instruct",
    "qwen2.5:72b":         "unsloth/Qwen2.5-72B-Instruct",
    # Qwen 3
    "qwen3:1.7b":          "unsloth/Qwen3-1.7B",
    "qwen3:4b":            "unsloth/Qwen3-4B",
    "qwen3:8b":            "unsloth/Qwen3-8B",
    "qwen3:14b":           "unsloth/Qwen3-14B",
    "qwen3:32b":           "unsloth/Qwen3-32B",
    # Gemma 2
    "gemma2:2b":           "unsloth/gemma-2-2b-it",
    "gemma2:9b":           "unsloth/gemma-2-9b-it",
    "gemma2:27b":          "unsloth/gemma-2-27b-it",
    # Gemma 3
    "gemma3:1b":           "unsloth/gemma-3-1b-it",
    "gemma3:4b":           "unsloth/gemma-3-4b-it",
    "gemma3:12b":          "unsloth/gemma-3-12b-it",
    "gemma3:27b":          "unsloth/gemma-3-27b-it",
    # Qwen 2.5 Coder
    "qwen2.5-coder:1.5b":  "unsloth/Qwen2.5-Coder-1.5B-Instruct",
    "qwen2.5-coder:7b":    "unsloth/Qwen2.5-Coder-7B-Instruct",
    "qwen2.5-coder:14b":   "unsloth/Qwen2.5-Coder-14B-Instruct",
    "qwen2.5-coder:14b-instruct": "unsloth/Qwen2.5-Coder-14B-Instruct",
    "qwen2.5-coder:32b":   "unsloth/Qwen2.5-Coder-32B-Instruct",
    # DeepSeek
    "deepseek-r1:7b":      "unsloth/DeepSeek-R1-Distill-Qwen-7B",
    "deepseek-r1:14b":     "unsloth/DeepSeek-R1-Distill-Qwen-14B",
    "deepseek-r1:32b":     "unsloth/DeepSeek-R1-Distill-Qwen-32B",
}


def _ollama_to_hf(name: str) -> str | None:
    import re
    key = name.lower().removesuffix(":latest")
    if key in _OLLAMA_HF_MAP:
        return _OLLAMA_HF_MAP[key]
    # For quantized variants (e.g. "deepseek-r1:32b-qwen-distill-q4_k_m"),
    # try matching just the model-name:size prefix.
    if ":" in key:
        model_name, tag = key.split(":", 1)
        m = re.match(r"(\d+(?:\.\d+)?[bm])", tag)
        if m:
            short_key = f"{model_name}:{m.group(1)}"
            if short_key in _OLLAMA_HF_MAP:
                return _OLLAMA_HF_MAP[short_key]
    return None


def _hf_cached(hf_model_id: str) -> bool:
    """Return True if the HuggingFace model weights are in the local cache."""
    cache_root = Path(os.environ.get("HF_HOME",
                      Path.home() / ".cache" / "huggingface")) / "hub"
    model_dir = cache_root / f"models--{hf_model_id.replace('/', '--')}"
    try:
        return model_dir.exists() and any(model_dir.iterdir())
    except OSError:
        return False


# ── Local (Unsloth) status ───────────────────────────────────────────────────

@router.get("/local-status", response_model=schemas.LocalStatusOut)
async def local_status(project_id: str):
    from ..local_trainer import check_unsloth
    return schemas.LocalStatusOut(**check_unsloth())


@router.get("/ollama-models")
async def ollama_models(project_id: str):
    """List installed Ollama models with their HuggingFace training equivalents."""
    import httpx
    from urllib.parse import urlparse
    _host = os.environ.get("OLLAMA_HOST", "")
    if not _host:
        base_url = "http://localhost:11434"
    else:
        if not _host.startswith(("http://", "https://")):
            _host = f"http://{_host}"
        _parsed = urlparse(_host)
        if not _parsed.port:
            _host = f"{_parsed.scheme}://{_parsed.hostname}:11434"
        base_url = _host.replace("0.0.0.0", "localhost").replace("[::]", "localhost")
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{base_url}/api/tags")
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:  # noqa: BLE001
        return {"models": [], "error": str(exc)}

    out = []
    for m in data.get("models", []):
        name: str = m.get("name", "")
        hf_id = _ollama_to_hf(name)
        out.append({
            "ollama_name": name,
            "hf_model": hf_id,
            "cached": _hf_cached(hf_id) if hf_id else False,
            "size_gb": round(m.get("size", 0) / 1e9, 1),
            "mapped": hf_id is not None,
        })

    # Sort: mapped first, then by size
    out.sort(key=lambda x: (not x["mapped"], x["size_gb"]))
    return {"models": out, "error": None}


# ── RunPod connectivity ──────────────────────────────────────────────────────

@router.get("/runpod-status", response_model=schemas.RunPodStatusOut)
async def runpod_status(project_id: str):
    endpoint_id = os.environ.get("RUNPOD_ENDPOINT_ID", "")
    health = await health_check()
    return schemas.RunPodStatusOut(
        configured=bool(os.environ.get("RUNPOD_API_KEY")) and bool(endpoint_id),
        endpoint_id=endpoint_id,
        health=health,
    )


@router.get("/gpu-types")
async def gpu_types(project_id: str):
    try:
        return {"gpu_types": await list_gpu_types()}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(503, str(exc))


# ── Training jobs ────────────────────────────────────────────────────────────

@router.get("/jobs", response_model=list[schemas.TrainingJobOut])
async def list_jobs(project_id: str, db: AsyncSession = Depends(get_session)):
    result = await db.execute(
        select(models.TrainingJob)
        .where(models.TrainingJob.project_id == project_id)
        .order_by(models.TrainingJob.created_at.desc())
    )
    return [_job_out(j) for j in result.scalars().all()]


@router.post("/start", response_model=schemas.TrainingJobOut)
async def start_training(project_id: str, cfg: schemas.TrainingConfig,
                         db: AsyncSession = Depends(get_session)):
    project = await db.get(models.Project, project_id)
    if not project:
        raise HTTPException(404, "Project not found")

    jsonl_text, sample_count = await _build_dataset(project_id, db, cfg)

    if cfg.provider == "local":
        return await _start_local(project_id, cfg, jsonl_text, sample_count, db)
    else:
        return await _start_runpod(project_id, cfg, jsonl_text, sample_count, db)


async def _start_local(project_id: str, cfg: schemas.TrainingConfig,
                       jsonl_text: str, sample_count: int,
                       db: AsyncSession) -> schemas.TrainingJobOut:
    from ..local_trainer import run_local_training

    job = models.TrainingJob(
        project_id=project_id,
        runpod_job_id=None,
        status="queued",
        config_json=json.dumps(cfg.model_dump()),
        log=(f"[{_now().isoformat()}] Local training queued.\n"
             f"Samples: {sample_count} | Base model: {cfg.base_model}\n"
             f"GGUF quantisation: {cfg.gguf_quantization}"
             + (f" | Ollama name: {cfg.ollama_model_name}" if cfg.ollama_model_name else "") + "\n"),
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)

    # Launch training in a thread — returns immediately, thread updates DB directly.
    asyncio.create_task(
        asyncio.to_thread(run_local_training, job.id, jsonl_text, cfg.model_dump(), EXPORT_DIR)
    )
    return _job_out(job)


async def _start_runpod(project_id: str, cfg: schemas.TrainingConfig,
                        jsonl_text: str, sample_count: int,
                        db: AsyncSession) -> schemas.TrainingJobOut:
    dataset_b64 = base64.b64encode(jsonl_text.encode()).decode()
    payload = {
        "dataset_b64": dataset_b64,
        "config": cfg.model_dump(),
        "sample_count": sample_count,
    }
    try:
        result = await submit_job(payload)
    except ValueError as exc:
        raise HTTPException(503, str(exc))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(502, f"RunPod error: {exc}")

    runpod_job_id = result.get("id")
    job = models.TrainingJob(
        project_id=project_id,
        runpod_job_id=runpod_job_id,
        status="queued",
        config_json=json.dumps(cfg.model_dump()),
        log=(f"[{_now().isoformat()}] Job submitted to RunPod. job_id={runpod_job_id}\n"
             f"Samples: {sample_count} | Base model: {cfg.base_model}\n"),
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)
    return _job_out(job)


@router.get("/jobs/{job_id}", response_model=schemas.TrainingJobOut)
async def get_job(project_id: str, job_id: str, db: AsyncSession = Depends(get_session)):
    job = await db.get(models.TrainingJob, job_id)
    if not job or job.project_id != project_id:
        raise HTTPException(404, "Training job not found")

    # Only poll RunPod for RunPod jobs that are still active.
    if job.runpod_job_id and job.status in ("queued", "running", "IN_QUEUE", "IN_PROGRESS"):
        try:
            rp = await job_status(job.runpod_job_id)
            _sync_runpod_status(job, (rp.get("status") or "").upper(), rp)
            await db.commit()
            await db.refresh(job)
        except Exception as exc:  # noqa: BLE001
            job.log += f"\n[{_now().isoformat()}] Status poll error: {exc}"
            await db.commit()

    return _job_out(job)


def _sync_runpod_status(job: models.TrainingJob, rp_status: str, rp: dict) -> None:
    status_map = {
        "IN_QUEUE": "queued", "IN_PROGRESS": "running",
        "COMPLETED": "completed", "FAILED": "failed",
        "CANCELLED": "cancelled", "TIMED_OUT": "failed",
    }
    new_status = status_map.get(rp_status, job.status)
    if new_status != job.status:
        job.log += f"\n[{_now().isoformat()}] Status → {new_status}"
        job.status = new_status

    output = rp.get("output") or {}
    if isinstance(output, dict):
        if output.get("log"):
            job.log += f"\n{output['log']}"
        if new_status == "completed" and output.get("adapter_repo"):
            # Worker uploaded the adapter to a (private) HF repo — download it.
            out_dir = Path(EXPORT_DIR) / "models" / job.id
            out_dir.mkdir(parents=True, exist_ok=True)
            repo_id = output["adapter_repo"]
            try:
                from huggingface_hub import snapshot_download
                token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None
                snapshot_download(repo_id=repo_id, repo_type="model",
                                  local_dir=str(out_dir), token=token)
                job.model_path = str(out_dir)
                job.log += f"\n[{_now().isoformat()}] Adapter downloaded from HF repo '{repo_id}' → {out_dir}"
            except Exception as exc:  # noqa: BLE001
                job.log += (f"\n[{_now().isoformat()}] Adapter is in HF repo '{repo_id}' but "
                            f"download failed: {exc}. Set HF_TOKEN in backend/.env to fetch it.")
        elif new_status == "completed" and output.get("model_files"):
            out_dir = Path(EXPORT_DIR) / "models" / job.id
            out_dir.mkdir(parents=True, exist_ok=True)
            for fname, b64_content in output["model_files"].items():
                (out_dir / fname).write_bytes(base64.b64decode(b64_content))
            job.model_path = str(out_dir)
            job.log += f"\n[{_now().isoformat()}] Adapter saved to {out_dir}"

    if new_status in ("completed", "failed", "cancelled") and not job.finished_at:
        job.finished_at = _now()


@router.post("/jobs/{job_id}/cancel")
async def cancel_training(project_id: str, job_id: str, db: AsyncSession = Depends(get_session)):
    job = await db.get(models.TrainingJob, job_id)
    if not job or job.project_id != project_id:
        raise HTTPException(404, "Training job not found")

    cfg = load_json(job.config_json, {})
    if cfg.get("provider") == "local" or not job.runpod_job_id:
        from ..local_trainer import request_cancel
        request_cancel(job.id)
        # Status will be updated by the thread; mark as cancelling now for the UI.
        job.log += f"\n[{_now().isoformat()}] Cancel signal sent — waiting for training step to complete."
    else:
        try:
            await cancel_job(job.runpod_job_id)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(502, f"RunPod cancel error: {exc}")
        job.status = "cancelled"
        job.finished_at = _now()
        job.log += f"\n[{_now().isoformat()}] Cancelled by user."

    await db.commit()
    return {"ok": True}


@router.get("/jobs/{job_id}/download")
async def download_model(project_id: str, job_id: str, db: AsyncSession = Depends(get_session)):
    job = await db.get(models.TrainingJob, job_id)
    if not job or job.project_id != project_id:
        raise HTTPException(404, "Training job not found")
    if not job.model_path or not Path(job.model_path).exists():
        raise HTTPException(404, "Model files not available yet.")

    import shutil, tempfile
    tmp = tempfile.mktemp(suffix=".zip")
    shutil.make_archive(tmp.removesuffix(".zip"), "zip", job.model_path)
    return FileResponse(tmp, filename=f"adapter_{job_id[:8]}.zip", media_type="application/zip")
