"""Dataset export in Alpaca / ShareGPT / OpenAI JSONL formats (spec 5.5).

Writes train/val JSONL files plus a manifest into the per-project export
directory. MVP target format is Alpaca; ShareGPT and OpenAI are included as
straightforward variants on the same row shape.
"""
from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path

from .config import EXPORT_DIR


def to_alpaca(sample) -> dict:
    return {
        "instruction": sample.instruction,
        "input": sample.input,
        "output": sample.output,
    }


def to_sharegpt(sample) -> dict:
    human = sample.instruction
    if sample.input:
        human = f"{sample.instruction}\n\n{sample.input}"
    return {
        "conversations": [
            {"from": "human", "value": human},
            {"from": "gpt", "value": sample.output},
        ]
    }


def to_openai(sample) -> dict:
    user = sample.instruction
    if sample.input:
        user = f"{sample.instruction}\n\n{sample.input}"
    return {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": sample.output},
        ]
    }


FORMATTERS = {
    "alpaca": to_alpaca,
    "sharegpt": to_sharegpt,
    "openai": to_openai,
}


def _stratified_split(samples: list, train_split: float) -> tuple[list, list]:
    """Split by sample type so each type keeps the train/val ratio (spec E7)."""
    by_type: dict[str, list] = {}
    for s in samples:
        by_type.setdefault(s.type, []).append(s)

    train: list = []
    val: list = []
    rng = random.Random(42)
    for rows in by_type.values():
        rows = rows[:]
        rng.shuffle(rows)
        cut = round(len(rows) * train_split)
        train.extend(rows[:cut])
        val.extend(rows[cut:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def export_dataset(project, samples: list, fmt: str, train_split: float,
                   include_statuses: list[str], manifest_extra: dict) -> dict:
    fmt = fmt.lower()
    formatter = FORMATTERS.get(fmt)
    if not formatter:
        raise ValueError(f"Unsupported export format: {fmt}")

    selected = [s for s in samples if s.status in include_statuses]
    train, val = _stratified_split(selected, train_split)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_dir = Path(EXPORT_DIR) / f"{project.id}" / f"{fmt}-{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "val.jsonl"
    manifest_path = out_dir / "manifest.json"

    def write_jsonl(path: Path, rows: list) -> None:
        with path.open("w", encoding="utf-8") as fh:
            for s in rows:
                fh.write(json.dumps(formatter(s), ensure_ascii=False) + "\n")

    write_jsonl(train_path, train)
    write_jsonl(val_path, val)

    by_type: dict[str, int] = {}
    for s in selected:
        by_type[s.type] = by_type.get(s.type, 0) + 1

    manifest = {
        "project": project.name,
        "project_id": project.id,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "format": fmt,
        "train_split": train_split,
        "include_statuses": include_statuses,
        "counts": {
            "total": len(selected),
            "train": len(train),
            "val": len(val),
            "by_type": by_type,
        },
        **manifest_extra,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {
        "manifest": manifest,
        "train_file": str(train_path),
        "val_file": str(val_path),
        "manifest_file": str(manifest_path),
        "train_count": len(train),
        "val_count": len(val),
    }
