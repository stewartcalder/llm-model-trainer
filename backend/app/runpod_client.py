"""RunPod API client — GraphQL for GPU/pod queries, REST for serverless jobs.

Docs: https://docs.runpod.io/serverless/references/
"""
from __future__ import annotations

import os
from typing import Any

import httpx

GRAPHQL_URL = "https://api.runpod.io/graphql"
REST_BASE = "https://api.runpod.io/v2"


def _api_key() -> str:
    key = os.environ.get("RUNPOD_API_KEY", "")
    if not key:
        raise ValueError("RUNPOD_API_KEY not set in environment / .env")
    return key


async def _gql(query: str, variables: dict | None = None) -> dict:
    key = _api_key()
    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.post(
            GRAPHQL_URL,
            json={"query": query, "variables": variables or {}},
            headers={"Authorization": f"Bearer {key}"},
        )
        resp.raise_for_status()
        body = resp.json()
        if "errors" in body:
            raise RuntimeError(body["errors"][0]["message"])
        return body["data"]


async def _rest(method: str, path: str, body: dict | None = None) -> Any:
    key = _api_key()
    endpoint_id = os.environ.get("RUNPOD_ENDPOINT_ID", "")
    url = f"{REST_BASE}/{endpoint_id}{path}"
    async with httpx.AsyncClient(timeout=60.0) as client:
        fn = getattr(client, method.lower())
        kwargs: dict = {"headers": {"Authorization": f"Bearer {key}"}}
        if body is not None:
            kwargs["json"] = body
        resp = await fn(url, **kwargs)
        resp.raise_for_status()
        return resp.json()


# ── GPU types ───────────────────────────────────────────────────────────────

async def list_gpu_types() -> list[dict]:
    data = await _gql("""
    query GpuTypes {
      gpuTypes {
        id
        displayName
        memoryInGb
        secureCloud
        communityCloud
        lowestPrice(input: { gpuCount: 1 }) {
          minimumBidPrice
          uninterruptablePrice
        }
      }
    }
    """)
    return data.get("gpuTypes", [])


# ── Serverless jobs ─────────────────────────────────────────────────────────

async def submit_job(payload: dict) -> dict:
    """Submit an async job to the configured serverless endpoint."""
    return await _rest("post", "/run", {"input": payload})


async def job_status(job_id: str) -> dict:
    return await _rest("get", f"/status/{job_id}")


async def cancel_job(job_id: str) -> dict:
    return await _rest("post", f"/cancel/{job_id}")


async def health_check() -> dict:
    """Check whether the configured endpoint is responsive."""
    endpoint_id = os.environ.get("RUNPOD_ENDPOINT_ID", "")
    if not endpoint_id:
        return {"ok": False, "detail": "RUNPOD_ENDPOINT_ID not set"}
    try:
        data = await _rest("get", "/health")
        return {"ok": True, "workers": data.get("workers", {}), "jobs": data.get("jobs", {})}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "detail": str(exc)}
