"""Connectivity probe for the configured LLM provider.

Returns a lightweight status dict that the frontend polls to show the
sidebar indicator. Intentionally does a real call where practical so the
status reflects actual reachability, not just "key is set".
"""
from __future__ import annotations

import os
import time


async def check_llm(llm_cfg: dict) -> dict:
    """Return {ok, provider, model, latency_ms, detail}."""
    provider = (llm_cfg.get("provider") or "mock").lower()
    model = llm_cfg.get("model", "")
    t0 = time.monotonic()

    try:
        if provider == "mock":
            return _ok(provider, model, t0, "offline mock — no API key needed")

        if provider == "anthropic":
            return await _check_anthropic(llm_cfg, model, t0)

        if provider in ("ollama", "openai", "openai_compat"):
            return await _check_openai_compat(llm_cfg, model, t0)

        return _err(provider, model, f"Unknown provider: {provider}")
    except Exception as exc:  # noqa: BLE001
        return _err(provider, model, str(exc))


def _ok(provider: str, model: str, t0: float, detail: str = "") -> dict:
    return {
        "ok": True,
        "provider": provider,
        "model": model,
        "latency_ms": round((time.monotonic() - t0) * 1000),
        "detail": detail,
    }


def _err(provider: str, model: str, detail: str) -> dict:
    return {"ok": False, "provider": provider, "model": model, "latency_ms": 0, "detail": detail}


async def _check_anthropic(cfg: dict, model: str, t0: float) -> dict:
    api_key = cfg.get("api_key") or os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return _err("anthropic", model, "ANTHROPIC_API_KEY not set")
    # Send a minimal message to verify the key is accepted.
    from anthropic import AsyncAnthropic, AuthenticationError, APIStatusError
    client = AsyncAnthropic(api_key=api_key)
    try:
        await client.messages.create(
            model=model,
            max_tokens=1,
            messages=[{"role": "user", "content": "ping"}],
        )
        return _ok("anthropic", model, t0)
    except AuthenticationError:
        return _err("anthropic", model, "Invalid API key")
    except APIStatusError as e:
        # 529 overloaded etc. — key is valid, service reachable
        if e.status_code in (429, 529):
            return _ok("anthropic", model, t0, f"HTTP {e.status_code} — key valid but rate-limited")
        return _err("anthropic", model, f"API error {e.status_code}: {e.message}")


async def _check_openai_compat(cfg: dict, model: str, t0: float) -> dict:
    """Ping the /models endpoint which is lightweight and doesn't cost tokens."""
    import httpx

    base = (cfg.get("base_url") or "http://localhost:11434/v1").rstrip("/")
    api_key = cfg.get("api_key") or os.environ.get("OPENAI_API_KEY", "ollama")
    headers = {"Authorization": f"Bearer {api_key}"}

    async with httpx.AsyncClient(timeout=8.0) as client:
        resp = await client.get(f"{base}/models", headers=headers)
        if resp.status_code == 200:
            data = resp.json()
            ids = [m.get("id", "") for m in data.get("data", [])]
            connected = model in ids
            detail = f"model {'found' if connected else 'NOT found'} on server"
            if not connected:
                detail += f" — available: {', '.join(ids[:5]) or 'none'}"
            return _ok(cfg.get("provider", "ollama"), model, t0, detail)
        return _err(cfg.get("provider", "ollama"), model,
                    f"Server returned HTTP {resp.status_code}")
