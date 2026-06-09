"""Local Unsloth LoRA trainer — runs in a thread, updates DB directly via psycopg2."""
from __future__ import annotations

import json
import subprocess
import traceback
from datetime import datetime, timezone
from pathlib import Path
from threading import Event

# Per-job cancellation flags.
_cancel_flags: dict[str, Event] = {}


def request_cancel(job_id: str) -> None:
    if job_id in _cancel_flags:
        _cancel_flags[job_id].set()


def check_unsloth() -> dict:
    try:
        import unsloth  # noqa: F401
        import torch
        version = getattr(unsloth, "__version__", "unknown")
        gpu = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if gpu else None
        return {
            "available": True,
            "version": version,
            "gpu": gpu,
            "detail": f"unsloth {version} · GPU: {gpu_name or '✗ none (CPU only — very slow)'}",
        }
    except ImportError as exc:
        return {
            "available": False,
            "version": None,
            "gpu": False,
            "detail": str(exc),
        }


# ── llama.cpp setup ──────────────────────────────────────────────────────────

def _ensure_llama_cpp(log_fn) -> None:
    """Ensure llama.cpp is built at ~/.unsloth/llama.cpp.

    Unsloth calls ``input()`` to confirm a system-package install when running
    outside Colab/Kaggle.  We patch ``builtins.input`` to auto-accept so the
    build can proceed unattended.  cmake must already be installed on the host
    (we install it as part of the server setup).
    """
    import builtins
    from pathlib import Path as _Path

    llama_dir = _Path.home() / ".unsloth" / "llama.cpp"
    if llama_dir.exists():
        return  # already installed

    log_fn("llama.cpp not found — building it now (one-time setup, ~10 min)…")
    log_fn("  cmake, git and gcc are required; cmake was installed as part of setup.")

    _orig_input = builtins.input

    def _auto_accept(prompt: str = "") -> str:
        log_fn(f"  [auto-accept] {prompt}")
        return ""  # same as pressing ENTER

    builtins.input = _auto_accept
    try:
        from unsloth_zoo.llama_cpp import install_llama_cpp  # noqa: PLC0415
        install_llama_cpp(print_output=False)
        log_fn("llama.cpp built successfully.")
    finally:
        builtins.input = _orig_input


# ── DB helpers (sync, psycopg2) ──────────────────────────────────────────────

def _conn():
    import psycopg2
    from .config import DATABASE_URL
    url = DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")
    return psycopg2.connect(url)


def _update_job(job_id: str, *,
                status: str | None = None,
                log_append: str | None = None,
                model_path: str | None = None,
                finished: bool = False) -> None:
    parts: list[str] = []
    vals: list = []
    if status is not None:
        parts.append("status = %s"); vals.append(status)
    if log_append is not None:
        parts.append("log = log || %s"); vals.append(log_append)
    if model_path is not None:
        parts.append("model_path = %s"); vals.append(model_path)
    if finished:
        parts.append("finished_at = %s"); vals.append(datetime.now(timezone.utc))
    if not parts:
        return
    vals.append(job_id)
    c = _conn()
    try:
        with c.cursor() as cur:
            cur.execute(f"UPDATE training_jobs SET {', '.join(parts)} WHERE id = %s", vals)
        c.commit()
    finally:
        c.close()


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("[%H:%M:%S]")


# ── Dataset formatting ────────────────────────────────────────────────────────

def _format_row(row: dict, fmt: str) -> str:
    if fmt == "sharegpt":
        return "\n".join(
            f"<{m.get('from','human')}> {m.get('value','')} </{m.get('from','human')}>"
            for m in row.get("conversations", [])
        )
    if fmt == "openai":
        return "\n".join(
            f"[{m.get('role','user')}] {m.get('content','')}"
            for m in row.get("messages", [])
        )
    # alpaca default
    parts = [f"### Instruction:\n{row.get('instruction', '')}"]
    if row.get("input"):
        parts.append(f"### Input:\n{row['input']}")
    parts.append(f"### Response:\n{row.get('output', '')}")
    return "\n".join(parts)


# ── Training entry point ──────────────────────────────────────────────────────

def run_local_training(job_id: str, dataset_jsonl: str, config: dict, export_dir: str) -> None:
    """Blocking. Call via asyncio.to_thread() from the FastAPI router."""
    cancel_evt = Event()
    _cancel_flags[job_id] = cancel_evt

    def log(msg: str) -> None:
        _update_job(job_id, log_append=f"\n{_ts()} {msg}")

    try:
        _update_job(job_id, status="running",
                    log_append=f"\n{_ts()} Local (Unsloth) training started.")

        log("Importing Unsloth…")
        from unsloth import FastLanguageModel, is_bfloat16_supported
        from datasets import Dataset
        from trl import SFTTrainer, SFTConfig
        from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

        class _CancelCallback(TrainerCallback):
            def on_log(self, args: TrainingArguments, state: TrainerState,
                       control: TrainerControl, logs: dict | None = None, **kwargs):
                if logs:
                    step = state.global_step
                    loss = logs.get("loss")
                    log(f"step {step}" + (f" — loss {loss:.4f}" if isinstance(loss, float) else ""))
                if cancel_evt.is_set():
                    log("Cancellation requested — stopping.")
                    control.should_training_stop = True
                return control

        base_model = config.get("base_model", "unsloth/Llama-3.2-1B-Instruct")
        max_seq_length = int(config.get("max_seq_length", 2048))
        use_4bit = bool(config.get("use_4bit", True))

        log(f"Loading base model: {base_model}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=use_4bit,
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r=int(config.get("lora_r", 16)),
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_alpha=int(config.get("lora_alpha", 32)),
            lora_dropout=float(config.get("lora_dropout", 0.05)),
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
        )

        rows = [json.loads(l) for l in dataset_jsonl.strip().splitlines() if l.strip()]
        fmt = config.get("dataset_format", "alpaca")
        dataset = Dataset.from_list([{"text": _format_row(r, fmt)} for r in rows])
        log(f"Dataset: {len(rows)} samples (format={fmt})")

        if cancel_evt.is_set():
            _update_job(job_id, status="cancelled", finished=True,
                        log_append=f"\n{_ts()} Cancelled before training.")
            return

        out_dir = Path(export_dir) / "models" / job_id
        ckpt_dir = out_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        bf16 = is_bfloat16_supported()
        sft_cfg = SFTConfig(
            output_dir=str(ckpt_dir),
            dataset_text_field="text",
            num_train_epochs=int(config.get("num_epochs", 3)),
            per_device_train_batch_size=int(config.get("batch_size", 4)),
            learning_rate=float(config.get("learning_rate", 2e-4)),
            max_seq_length=max_seq_length,
            bf16=bf16,
            fp16=not bf16,
            logging_steps=1,
            save_strategy="no",
            report_to="none",
        )

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            args=sft_cfg,
            callbacks=[_CancelCallback()],
        )

        log("Training…")
        trainer.train()

        if cancel_evt.is_set():
            _update_job(job_id, status="cancelled", finished=True,
                        log_append=f"\n{_ts()} Cancelled after training.")
            return

        # Save LoRA adapter files.
        adapter_dir = out_dir / "adapter"
        log(f"Saving LoRA adapter → {adapter_dir.name}/")
        model.save_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))

        # Merge and export to GGUF for Ollama.
        quant = config.get("gguf_quantization", "q4_k_m")
        gguf_dir = out_dir / "gguf"
        gguf_dir.mkdir(parents=True, exist_ok=True)
        log(f"Merging weights and exporting GGUF ({quant}) — this can take several minutes…")
        _ensure_llama_cpp(log)
        # Point HuggingFace at the GLOBAL cache so the fp16 base-model download
        # (needed for GGUF conversion) is cached across training runs rather than
        # being re-downloaded into each job's output directory.
        import os as _os
        _hf_home = str(Path.home() / ".cache" / "huggingface")
        _os.environ["HF_HOME"] = _hf_home
        _os.environ["HUGGINGFACE_HUB_CACHE"] = str(Path(_hf_home) / "hub")
        log(f"HF cache → {_hf_home}")
        model.save_pretrained_gguf(str(gguf_dir), tokenizer, quantization_method=quant)

        # Unsloth appends "_gguf" to the save directory; search there first,
        # then fall back to a full recursive scan of the job output tree.
        gguf_actual_dir = Path(str(gguf_dir) + "_gguf")
        gguf_files = (list(gguf_actual_dir.glob("*.gguf"))
                      if gguf_actual_dir.exists() else [])
        if not gguf_files:
            gguf_files = list(out_dir.rglob("*.gguf"))
        if not gguf_files:
            raise RuntimeError("GGUF export produced no .gguf file.")
        gguf_path = gguf_files[0]
        size_mb = gguf_path.stat().st_size // (1024 * 1024)
        log(f"GGUF ready: {gguf_path.name} ({size_mb} MB)")

        # Register with local Ollama if the user provided a name.
        ollama_name = (config.get("ollama_model_name") or "").strip()
        if ollama_name:
            # Prefer Unsloth's generated Modelfile (has full chat template),
            # but fix the FROM line to an absolute path so ollama create works
            # from any working directory.
            modelfile = gguf_path.parent / "Modelfile"
            if modelfile.exists():
                lines = modelfile.read_text().splitlines()
                lines = [f"FROM {gguf_path.absolute()}" if l.startswith("FROM ") else l
                         for l in lines]
                modelfile.write_text("\n".join(lines) + "\n")
            else:
                modelfile.write_text(f"FROM {gguf_path.absolute()}\n")
            log(f"Running: ollama create {ollama_name}")
            proc = subprocess.run(
                ["ollama", "create", ollama_name, "-f", str(modelfile)],
                capture_output=True, text=True, timeout=600,
            )
            if proc.returncode != 0:
                raise RuntimeError(f"ollama create failed: {proc.stderr.strip()}")
            log(f"Ollama model '{ollama_name}' is ready — try: ollama run {ollama_name}")
        else:
            log(f"No Ollama name set. GGUF is at: {gguf_path}")

        _update_job(job_id, status="completed", model_path=str(out_dir), finished=True,
                    log_append=f"\n{_ts()} Done.")

    except Exception:
        _update_job(job_id, status="failed", finished=True,
                    log_append=f"\n{_ts()} ERROR:\n{traceback.format_exc()}")
    finally:
        _cancel_flags.pop(job_id, None)
