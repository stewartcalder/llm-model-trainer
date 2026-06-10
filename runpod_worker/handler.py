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

import base64, json, os, sys, tempfile, traceback
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

    return output_dir, "\n".join(
        [f"step {e['step']}: loss={e.get('loss', '?')}" for e in trainer.state.log_history if "loss" in e]
    )


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

        # ── Return the adapter ───────────────────────────────────────────────
        # Large adapters (e.g. 14B LoRA ≈ 140 MB → ~190 MB base64) can exceed
        # RunPod's job-output size limit. If an HF write token + target repo are
        # configured on the endpoint, upload there and return the repo id instead
        # of inlining the bytes. Otherwise fall back to base64 (fine for ≤7B).
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        # Repo can be set per-job (config.hf_upload_repo) or per-endpoint (env).
        hf_repo = cfg.get("hf_upload_repo") or os.environ.get("HF_UPLOAD_REPO")

        total_mb = sum(
            f.stat().st_size for f in Path(output_dir).iterdir() if f.is_file()
        ) / (1024 * 1024)
        print(f"[handler] adapter size: {total_mb:.1f} MB")

        if hf_token and hf_repo:
            from huggingface_hub import HfApi
            api = HfApi(token=hf_token)
            print(f"[handler] uploading adapter to HF repo '{hf_repo}' (private)…")
            api.create_repo(hf_repo, private=True, exist_ok=True, repo_type="model")
            api.upload_folder(
                folder_path=output_dir,
                repo_id=hf_repo,
                repo_type="model",
                commit_message="LoRA adapter (trained on RunPod)",
            )
            print("[handler] upload complete")
            return {"adapter_repo": hf_repo, "adapter_size_mb": round(total_mb, 1), "log": log}

        # Fallback: base64-encode adapter files into the job output.
        if total_mb > 50:
            print(
                f"[handler] WARNING: adapter is {total_mb:.0f} MB and no HF_TOKEN/"
                "HF_UPLOAD_REPO is set — base64 return may exceed RunPod's output "
                "limit. Set HF_TOKEN + HF_UPLOAD_REPO on the endpoint to avoid this."
            )
        model_files: dict[str, str] = {}
        for fpath in Path(output_dir).iterdir():
            if fpath.is_file():
                model_files[fpath.name] = base64.b64encode(fpath.read_bytes()).decode()

        return {"model_files": model_files, "adapter_size_mb": round(total_mb, 1), "log": log}

    except Exception as exc:
        tb = traceback.format_exc()
        print(f"[handler] ERROR:\n{tb}", file=sys.stderr)
        return {"error": str(exc), "traceback": tb}


runpod.serverless.start({"handler": handler})
