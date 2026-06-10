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

        # ── GGUF export (zero HuggingFace network traffic) ──────────────────
        # Strategy: use PEFT's merge_and_unload() which dequantizes the 4-bit
        # weights already resident in VRAM and merges the LoRA deltas in-place.
        # The resulting full-precision model is saved with transformers'
        # save_pretrained (writes the in-memory state dict — no download).
        # This entirely avoids the ~15 GB fp16 re-download that both
        # save_pretrained_gguf and save_pretrained_merged(merged_16bit) trigger.
        import gc
        quant = config.get("gguf_quantization", "q4_k_m")
        merged_dir = out_dir / "merged"
        gguf_out   = out_dir / "gguf_out"
        merged_dir.mkdir(parents=True, exist_ok=True)
        gguf_out.mkdir(parents=True, exist_ok=True)

        # ── Merge LoRA + stream-write shards (low RAM) ──────────────────────────
        # PEFT's merge_and_unload() leaves bnb tensors (uint8 packed weights,
        # .absmax) in the state dict — GGUF converter rejects these.  Unsloth's
        # internal _merge_lora() calls fast_dequantize on each Bnb_Linear4bit
        # layer, producing clean fp16/bf16 tensors.  We replicate the loop from
        # unsloth/save.py and flush each shard to disk immediately (≤3 GB) so we
        # never accumulate the full 14 GB in RAM before writing — which would push
        # a 15 GB machine into swap and stall for hours.
        import torch
        import json as _json
        import os as _os
        from safetensors.torch import save_file as _sf_save
        from peft import PeftModelForCausalLM
        from unsloth.save import _merge_lora, LLAMA_WEIGHTS, LLAMA_LAYERNORMS

        if isinstance(model, PeftModelForCausalLM):
            internal_model = model.model
        else:
            internal_model = model

        cfg_dtype = getattr(internal_model.config, "torch_dtype", "bfloat16")
        torch_dtype = torch.bfloat16 if "bfloat16" in str(cfg_dtype) else torch.float16
        has_lm_head = (
            internal_model.model.embed_tokens.weight.data_ptr()
            != internal_model.lm_head.weight.data_ptr()
        )

        # Capture config before the merge loop frees GPU state
        cfg_dict = internal_model.config.to_dict()
        cfg_dict.pop("quantization_config", None)

        SHARD_BYTES = 3 * 1024 ** 3   # flush every 3 GB — keeps peak RAM low
        cur_shard: dict = {}
        cur_bytes  = 0
        part_names: list = []
        weight_map: dict = {}
        total_bytes = 0

        def _flush():
            nonlocal cur_shard, cur_bytes
            if not cur_shard:
                return
            idx  = len(part_names)
            fname = f"model-part-{idx:05d}.safetensors"
            log(f"  writing shard {idx + 1} ({cur_bytes // 1024 ** 2} MB)…")
            _sf_save(cur_shard, str(merged_dir / fname))
            for k in cur_shard:
                weight_map[k] = fname
            part_names.append(fname)
            cur_shard = {}
            cur_bytes  = 0
            gc.collect()
            torch.cuda.empty_cache()

        def _add(name: str, tensor: torch.Tensor) -> None:
            nonlocal cur_bytes, total_bytes
            t = tensor.contiguous().cpu()
            cur_shard[name] = t
            cur_bytes  += t.nbytes
            total_bytes += t.nbytes
            if cur_bytes >= SHARD_BYTES:
                _flush()

        log("Merging LoRA layer-by-layer and streaming shards to disk…")
        _add("model.embed_tokens.weight",
             internal_model.model.embed_tokens.weight.data.to(torch_dtype))

        n_layers = len(internal_model.model.layers)
        for j, layer in enumerate(internal_model.model.layers):
            if j % 4 == 0:
                log(f"  merging layers {j}–{min(j + 3, n_layers - 1)} / {n_layers}…")
            for item in LLAMA_WEIGHTS:
                try:
                    proj = eval(f"layer.{item}")  # noqa: S307
                    name = f"model.layers.{j}.{item}.weight"
                    W, bias = _merge_lora(proj, name)
                    _add(name, W.to(torch_dtype))
                    if bias is not None:
                        _add(f"model.layers.{j}.{item}.bias", bias)
                except Exception:
                    pass
            for item in LLAMA_LAYERNORMS:
                try:
                    _add(f"model.layers.{j}.{item}.weight",
                         eval(f"layer.{item}.weight.data"))  # noqa: S307
                except Exception:
                    continue

        _add("model.norm.weight", internal_model.model.norm.weight.data)
        if has_lm_head:
            _add("lm_head.weight",
                 internal_model.lm_head.weight.data.to(torch_dtype))

        _flush()   # write remaining tensors

        # Rename part files to standard HF shard naming
        n_parts = len(part_names)
        log(f"Finalising {n_parts} shard(s)…")
        if n_parts == 1:
            _os.rename(str(merged_dir / part_names[0]),
                       str(merged_dir / "model.safetensors"))
        else:
            final_map: dict = {}
            for i, old in enumerate(part_names):
                new = f"model-{i + 1:05d}-of-{n_parts:05d}.safetensors"
                _os.rename(str(merged_dir / old), str(merged_dir / new))
                for k, v in weight_map.items():
                    if v == old:
                        final_map[k] = new
            (merged_dir / "model.safetensors.index.json").write_text(
                _json.dumps(
                    {"metadata": {"total_size": total_bytes}, "weight_map": final_map},
                    indent=2,
                )
            )

        gc.collect()
        torch.cuda.empty_cache()

        # Config, generation_config, tokenizer
        (merged_dir / "config.json").write_text(_json.dumps(cfg_dict, indent=2))
        if hasattr(internal_model, "generation_config"):
            internal_model.generation_config.save_pretrained(str(merged_dir))
        _tok = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer
        _tok.save_pretrained(str(merged_dir))
        gc.collect()

        _ensure_llama_cpp(log)
        llama_dir = Path.home() / ".unsloth" / "llama.cpp"
        converter  = llama_dir / "convert_hf_to_gguf.py"
        quantizer  = llama_dir / "llama-quantize"

        # Step 1: convert merged safetensors → f16 GGUF
        # Use sys.executable so the converter runs in the same venv (has torch, transformers, etc.)
        import sys
        f16_gguf = gguf_out / "model-f16.gguf"
        log("Converting to GGUF (f16)…")
        proc = subprocess.run(
            [sys.executable, str(converter), str(merged_dir), "--outfile", str(f16_gguf), "--outtype", "f16"],
            capture_output=True, text=True, timeout=1800,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"convert_hf_to_gguf failed:\n{proc.stderr[-2000:]}")

        # Step 2: quantise f16 GGUF → target quantisation
        quant_gguf = gguf_out / f"model-{quant}.gguf"
        log(f"Quantising to {quant}…")
        proc = subprocess.run(
            [str(quantizer), str(f16_gguf), str(quant_gguf), quant.upper()],
            capture_output=True, text=True, timeout=1800,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"llama-quantize failed:\n{proc.stderr[-2000:]}")

        # Remove the intermediate f16 GGUF to save ~15 GB
        f16_gguf.unlink(missing_ok=True)

        gguf_path = quant_gguf
        size_mb = gguf_path.stat().st_size // (1024 * 1024)
        log(f"GGUF ready: {gguf_path.name} ({size_mb} MB)")

        # Register with local Ollama if the user provided a name.
        ollama_name = (config.get("ollama_model_name") or "").strip()
        if ollama_name:
            modelfile = gguf_out / "Modelfile"
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
