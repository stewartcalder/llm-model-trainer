"""
RunPod serverless handler for LoRA / QLoRA fine-tuning.

Expects input:
{
  "dataset_b64": "<base64-encoded JSONL string>",
  "config": {
    "base_model":     "meta-llama/Llama-3.2-1B",
    "lora_r":         16,
    "lora_alpha":     32,
    "lora_dropout":   0.05,
    "num_epochs":     3,
    "batch_size":     4,
    "learning_rate":  2e-4,
    "max_seq_length": 2048,
    "use_4bit":       true,
    "dataset_format": "alpaca"   // "alpaca" | "sharegpt" | "openai"
  }
}

Returns:
{
  "model_files": {
    "adapter_config.json": "<base64>",
    "adapter_model.safetensors": "<base64>",
    ...
  },
  "log": "<training log string>"
}
"""

import base64, json, os, subprocess, sys, tempfile, traceback
from pathlib import Path
import runpod

# ── lazy imports so Docker health-checks don't pay the GPU init cost ──────
def _import_training_libs():
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model
    from trl import SFTTrainer, SFTConfig
    return torch, Dataset, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, LoraConfig, get_peft_model, SFTTrainer, SFTConfig


def _fmt_alpaca(row: dict) -> str:
    parts = ["### Instruction:", row.get("instruction", "")]
    if row.get("input"):
        parts += ["### Input:", row["input"]]
    parts += ["### Response:", row.get("output", "")]
    return "\n".join(parts)


def _fmt_sharegpt(row: dict) -> str:
    result = []
    for msg in row.get("conversations", []):
        role = msg.get("from", "human")
        result.append(f"<{role}> {msg.get('value', '')} </{role}>")
    return "\n".join(result)


def _fmt_openai(row: dict) -> str:
    result = []
    for msg in row.get("messages", []):
        result.append(f"[{msg.get('role', 'user')}] {msg.get('content', '')}")
    return "\n".join(result)


def _load_dataset(jsonl_text: str, fmt: str):
    torch, Dataset, *_ = _import_training_libs()
    rows = [json.loads(l) for l in jsonl_text.strip().splitlines() if l.strip()]
    fmt_map = {"alpaca": _fmt_alpaca, "sharegpt": _fmt_sharegpt, "openai": _fmt_openai}
    formatter = fmt_map.get(fmt, _fmt_alpaca)
    return Dataset.from_list([{"text": formatter(r)} for r in rows])


def _train(cfg: dict, dataset) -> tuple[str, str]:
    torch, _, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, LoraConfig, get_peft_model, SFTTrainer, SFTConfig = _import_training_libs()

    base_model = cfg["base_model"]
    output_dir = tempfile.mkdtemp(prefix="lora_")

    bnb_cfg = None
    if cfg.get("use_4bit", True):
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_cfg,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    peft_cfg = LoraConfig(
        r=cfg.get("lora_r", 16),
        lora_alpha=cfg.get("lora_alpha", 32),
        lora_dropout=cfg.get("lora_dropout", 0.05),
        bias="none",
        task_type="CAUSAL_LM",
    )

    sft_cfg = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=cfg.get("num_epochs", 3),
        per_device_train_batch_size=cfg.get("batch_size", 4),
        learning_rate=cfg.get("learning_rate", 2e-4),
        max_seq_length=cfg.get("max_seq_length", 2048),
        fp16=not cfg.get("use_4bit", True),
        bf16=cfg.get("use_4bit", True),
        logging_steps=1,
        save_strategy="no",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_cfg,
        args=sft_cfg,
    )
    trainer.train()
    trainer.save_model(output_dir)

    log = "\n".join(
        f"step {e['step']}: loss={e.get('loss', '?')}"
        for e in trainer.state.log_history if "loss" in e
    )

    # Free the 4-bit training model before the fp16 merge reloads the base.
    del trainer, model
    torch.cuda.empty_cache()

    return output_dir, log


# ── GGUF export (runs on the RunPod GPU) ──────────────────────────────────────

_LLAMA_DIR = os.environ.get("LLAMA_CPP_DIR", "/opt/llama.cpp")


def _export_gguf(cfg: dict, adapter_dir: str, work_dir: str) -> str:
    """Merge the LoRA adapter into a fresh fp16 base, convert to GGUF and
    quantise. Returns the path to the quantised .gguf.

    Reloads the base model in fp16 (not 4-bit) so merge_and_unload yields clean
    fp16 weights — no bitsandbytes dequant artefacts. Intermediate files are
    deleted aggressively to stay within the container disk budget.
    """
    import shutil
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    base_model = cfg["base_model"]
    quant = cfg.get("gguf_quantization", "q4_k_m")
    merged_dir = os.path.join(work_dir, "merged")
    os.makedirs(merged_dir, exist_ok=True)

    print(f"[gguf] reloading base {base_model} in fp16 and merging adapter…")
    base = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
    merged.save_pretrained(merged_dir, safe_serialization=True, max_shard_size="5GB")
    AutoTokenizer.from_pretrained(base_model, trust_remote_code=True).save_pretrained(merged_dir)
    del merged, base
    torch.cuda.empty_cache()

    # Free the base model from the HF cache to reclaim ~disk before conversion.
    try:
        from huggingface_hub import scan_cache_dir
        for repo in scan_cache_dir().repos:
            if repo.repo_id == base_model:
                scan_cache_dir().delete_revisions(
                    *[rev.commit_hash for rev in repo.revisions]
                ).execute()
    except Exception as exc:  # noqa: BLE001
        print(f"[gguf] cache cleanup skipped: {exc}")

    converter = os.path.join(_LLAMA_DIR, "convert_hf_to_gguf.py")
    quantizer = os.path.join(_LLAMA_DIR, "build", "bin", "llama-quantize")
    f16_gguf = os.path.join(work_dir, "model-f16.gguf")
    out_gguf = os.path.join(work_dir, f"model-{quant}.gguf")

    print("[gguf] converting merged model → f16 GGUF…")
    subprocess.run(
        [sys.executable, converter, merged_dir, "--outfile", f16_gguf, "--outtype", "f16"],
        check=True,
    )
    shutil.rmtree(merged_dir, ignore_errors=True)   # free ~fp16 shards

    print(f"[gguf] quantising → {quant}…")
    subprocess.run([quantizer, f16_gguf, out_gguf, quant.upper()], check=True)
    os.remove(f16_gguf)                              # free the f16 intermediate

    size_mb = os.path.getsize(out_gguf) / (1024 * 1024)
    print(f"[gguf] ready: {os.path.basename(out_gguf)} ({size_mb:.0f} MB)")
    return out_gguf


def handler(event: dict) -> dict:
    job_input = event.get("input", {})
    try:
        dataset_b64 = job_input["dataset_b64"]
        cfg = job_input.get("config", {})

        jsonl_text = base64.b64decode(dataset_b64).decode("utf-8")
        fmt = cfg.get("dataset_format", "alpaca")

        print(f"[handler] loading dataset ({len(jsonl_text)} bytes, format={fmt})")
        dataset = _load_dataset(jsonl_text, fmt)
        print(f"[handler] dataset rows: {len(dataset)}")

        print(f"[handler] starting training: {cfg}")
        output_dir, log = _train(cfg, dataset)
        print(f"[handler] training done, saved to {output_dir}")

        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        # Repo can be set per-job (config.hf_upload_repo) or per-endpoint (env).
        hf_repo = cfg.get("hf_upload_repo") or os.environ.get("HF_UPLOAD_REPO")

        adapter_mb = sum(
            f.stat().st_size for f in Path(output_dir).iterdir() if f.is_file()
        ) / (1024 * 1024)
        print(f"[handler] adapter size: {adapter_mb:.1f} MB")

        result: dict = {"log": log, "adapter_size_mb": round(adapter_mb, 1)}

        # ── Upload the adapter (always — cheap insurance so a GGUF failure does
        #    not waste the training run). Needs HF token + repo. ──────────────
        api = None
        if hf_token and hf_repo:
            from huggingface_hub import HfApi
            api = HfApi(token=hf_token)
            print(f"[handler] uploading adapter to HF repo '{hf_repo}' (private)…")
            api.create_repo(hf_repo, private=True, exist_ok=True, repo_type="model")
            api.upload_folder(folder_path=output_dir, repo_id=hf_repo, repo_type="model",
                              path_in_repo="adapter", commit_message="LoRA adapter (RunPod)")
            result["adapter_repo"] = hf_repo

        # ── GGUF export on the GPU, then upload the .gguf (default on). ───────
        # The backend downloads the .gguf and runs `ollama create` locally, so a
        # RunPod run lands in Ollama exactly like a local build.
        if cfg.get("export_gguf", True) and api is not None:
            try:
                gguf_path = _export_gguf(cfg, output_dir, output_dir)
                gguf_name = os.path.basename(gguf_path)
                gguf_mb = os.path.getsize(gguf_path) / (1024 * 1024)
                print(f"[handler] uploading {gguf_name} ({gguf_mb:.0f} MB) to '{hf_repo}'…")
                api.upload_file(path_or_fileobj=gguf_path, path_in_repo=gguf_name,
                                repo_id=hf_repo, repo_type="model",
                                commit_message="GGUF (RunPod)")
                result.update({
                    "gguf_repo": hf_repo,
                    "gguf_filename": gguf_name,
                    "gguf_size_mb": round(gguf_mb, 1),
                    "ollama_model_name": cfg.get("ollama_model_name") or "",
                })
                result["log"] += f"\n[gguf] exported {gguf_name} ({gguf_mb:.0f} MB)"
            except Exception as exc:  # noqa: BLE001
                tb = traceback.format_exc()
                print(f"[handler] GGUF export failed (adapter still saved):\n{tb}", file=sys.stderr)
                result["gguf_error"] = str(exc)
                result["log"] += f"\n[gguf] export FAILED: {exc} (adapter is still in '{hf_repo}/adapter')"
            return result

        if api is not None:
            return result

        # ── No HF repo configured: base64 the adapter into the job output. ───
        if adapter_mb > 50:
            print(f"[handler] WARNING: adapter is {adapter_mb:.0f} MB and no HF_TOKEN/"
                  "HF_UPLOAD_REPO is set — base64 return may exceed RunPod's output limit.")
        model_files: dict[str, str] = {}
        for fpath in Path(output_dir).iterdir():
            if fpath.is_file():
                model_files[fpath.name] = base64.b64encode(fpath.read_bytes()).decode()
        result["model_files"] = model_files
        return result

    except Exception as exc:
        tb = traceback.format_exc()
        print(f"[handler] ERROR:\n{tb}", file=sys.stderr)
        return {"error": str(exc), "traceback": tb}


runpod.serverless.start({"handler": handler})
