"""LLM provider abstraction (spec 5.3.3 / 5.3.4).

Providers: ``mock`` (deterministic, offline, no key), ``anthropic`` (official
SDK), and ``ollama`` / OpenAI-compatible endpoints (via the openai SDK). Each
call returns generated text plus token usage for the cost tally. Retries use
exponential backoff (3 attempts).
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import re
from dataclasses import dataclass


@dataclass
class LLMResult:
    text: str
    input_tokens: int
    output_tokens: int


class LLMError(Exception):
    pass


def _extract_json(text: str) -> dict:
    """Pull the first JSON object out of a model response."""
    text = text.strip()
    # Strip ```json fences if present.
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidate = text[start : end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    raise LLMError(f"Response was not valid JSON: {text[:200]}")


class BaseProvider:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.model = cfg.get("model", "")
        self.temperature = float(cfg.get("temperature", 0.7))
        self.max_tokens = int(cfg.get("max_tokens", 1024))

    async def complete(self, prompt: str) -> LLMResult:  # pragma: no cover - abstract
        raise NotImplementedError


class MockProvider(BaseProvider):
    """Deterministic offline provider so the full pipeline is runnable without
    any API key. Produces plausible JSON for each prompt type."""

    async def complete(self, prompt: str) -> LLMResult:
        await asyncio.sleep(0.01)
        passage = prompt.split("PASSAGE:", 1)[-1]
        passage = passage.split("SOURCE:", 1)[0].strip()
        first = (passage.split(".")[0] or passage)[:160].strip()

        if "quality reviewer" in prompt:
            payload = {
                "faithfulness": 5,
                "completeness": 4,
                "clarity": 5,
                "reject": False,
                "reason": "Grounded in the source passage.",
            }
        elif '"question"' in prompt:
            payload = {
                "question": f"What does the passage explain about: {first}?",
                "answer": f"The passage explains that {first.lower() or 'the topic'}. "
                "It provides supporting detail drawn directly from the source material.",
            }
        else:
            payload = {
                "instruction": f"Explain the following concept: {first}.",
                "input": "",
                "output": f"{first}. This response summarises the key point grounded "
                "in the provided source material.",
            }
        text = json.dumps(payload)
        return LLMResult(text, len(prompt) // 4, len(text) // 4)


class AnthropicProvider(BaseProvider):
    async def complete(self, prompt: str) -> LLMResult:
        from anthropic import AsyncAnthropic

        api_key = self.cfg.get("api_key") or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise LLMError("ANTHROPIC_API_KEY not set (project settings or environment).")
        client = AsyncAnthropic(api_key=api_key)
        resp = await client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        return LLMResult(text, resp.usage.input_tokens, resp.usage.output_tokens)


class OpenAICompatProvider(BaseProvider):
    """Ollama and any OpenAI-compatible /v1/chat/completions endpoint."""

    async def complete(self, prompt: str) -> LLMResult:
        from openai import AsyncOpenAI

        base_url = self.cfg.get("base_url", "http://localhost:11434/v1")
        api_key = self.cfg.get("api_key") or os.environ.get("OPENAI_API_KEY") or "ollama"
        client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        resp = await client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.choices[0].message.content or ""
        usage = resp.usage
        in_tok = getattr(usage, "prompt_tokens", 0) if usage else 0
        out_tok = getattr(usage, "completion_tokens", 0) if usage else 0
        return LLMResult(text, in_tok, out_tok)


def make_provider(llm_cfg: dict) -> BaseProvider:
    provider = (llm_cfg.get("provider") or "mock").lower()
    if provider == "mock":
        return MockProvider(llm_cfg)
    if provider == "anthropic":
        return AnthropicProvider(llm_cfg)
    if provider in ("ollama", "openai", "openai_compat"):
        return OpenAICompatProvider(llm_cfg)
    raise LLMError(f"Unknown provider: {provider}")


async def complete_json(provider: BaseProvider, prompt: str, attempts: int = 3) -> tuple[dict, LLMResult]:
    """Call the provider and parse JSON, retrying with exponential backoff."""
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            result = await provider.complete(prompt)
            return _extract_json(result.text), result
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < attempts - 1:
                await asyncio.sleep((2**attempt) + random.random())
    raise LLMError(str(last_exc))
