"""Application configuration and default pipeline config."""
from __future__ import annotations

import os
from pathlib import Path

# Project data lives in <repo>/data by default. Source files, the SQLite DB,
# and exports all stay local (NFR: data residency).
DATA_DIR = Path(os.environ.get("LDB_DATA_DIR", Path(__file__).resolve().parents[2] / "data"))
UPLOAD_DIR = DATA_DIR / "uploads"
EXPORT_DIR = DATA_DIR / "exports"
DB_PATH = DATA_DIR / "app.db"

for _d in (DATA_DIR, UPLOAD_DIR, EXPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

DATABASE_URL = f"sqlite+aiosqlite:///{DB_PATH}"

# Default pipeline config applied to new projects (mirrors spec section 9).
DEFAULT_CONFIG: dict = {
    "chunking": {
        "strategy": "sentence_window",
        "window_tokens": 512,
        "overlap_tokens": 64,
    },
    "sample_types": ["qa", "instruction"],
    "llm": {
        "provider": "mock",  # mock | anthropic | ollama
        "model": "claude-haiku-4-5-20251001",
        "base_url": "http://localhost:11434/v1",  # used by ollama / openai-compatible
        "temperature": 0.7,
        "max_tokens": 1024,
        "use_critic": True,
    },
    "concurrency": 5,
    "budget_usd": 5.0,
    "export": {
        "format": "alpaca",
        "train_split": 0.9,
        "include_statuses": ["approved"],
    },
}

# Rough per-million-token USD prices for the cost estimator. These are only used
# for the running tally / budget guard and can be tuned per model.
PRICE_TABLE: dict[str, tuple[float, float]] = {
    # model substring : (input $/Mtok, output $/Mtok)
    "haiku": (0.80, 4.0),
    "sonnet": (3.0, 15.0),
    "opus": (15.0, 75.0),
}


def price_for_model(model: str) -> tuple[float, float]:
    m = (model or "").lower()
    for key, price in PRICE_TABLE.items():
        if key in m:
            return price
    return (1.0, 5.0)  # generic fallback
