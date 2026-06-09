"""FastAPI application entrypoint."""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .database import get_session, init_db
from .llm_status import check_llm
from .prompts import SAMPLE_TYPE_LABELS
from .routers import exports, projects, runs, samples, sources
from .routers import training
from .serialize import load_json

FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


app = FastAPI(title="LoRA Training Data Builder", version=__version__, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(projects.router)
app.include_router(sources.router)
app.include_router(samples.router)
app.include_router(runs.router)
app.include_router(exports.router)
app.include_router(training.router)


@app.get("/api/health")
async def health():
    return {"status": "ok", "version": __version__}


@app.get("/api/meta")
async def meta():
    return {
        "sample_types": [{"id": k, "label": v} for k, v in SAMPLE_TYPE_LABELS.items()],
        "providers": ["mock", "anthropic", "ollama"],
        "export_formats": ["alpaca", "sharegpt", "openai"],
    }


@app.get("/api/llm-status")
async def llm_status(project_id: str = Query(...)):
    """Probe the LLM configured for the given project and return connectivity info."""
    from sqlalchemy import select
    from . import models

    async for db in get_session():
        project = await db.get(models.Project, project_id)
        if not project:
            return {"ok": False, "provider": "", "model": "", "latency_ms": 0,
                    "detail": "Project not found"}
        llm_cfg = load_json(project.config_json, {}).get("llm", {})
    return await check_llm(llm_cfg)


# ---- Static frontend (production build) ----
if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")

    @app.get("/{full_path:path}")
    async def spa(full_path: str):
        candidate = FRONTEND_DIST / full_path
        if full_path and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(FRONTEND_DIST / "index.html")
