"""FastAPI application entrypoint.

Serves the JSON API under /api and the built React SPA (if present) as static
files, so the whole single-user app runs from one process.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .database import init_db
from .prompts import SAMPLE_TYPE_LABELS
from .routers import exports, projects, runs, samples, sources

FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


app = FastAPI(title="LoRA Training Data Builder", version=__version__, lifespan=lifespan)

# Permissive CORS so `vite dev` on :5173 can talk to the API during development.
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


# ---- Static frontend (production build) ----
if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")

    @app.get("/{full_path:path}")
    async def spa(full_path: str):
        # Serve real files when they exist, otherwise fall back to index.html
        # so client-side routing works.
        candidate = FRONTEND_DIST / full_path
        if full_path and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(FRONTEND_DIST / "index.html")
