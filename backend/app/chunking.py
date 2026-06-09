"""Sentence-window chunking with token counting (spec C1).

Tokenisation uses tiktoken when available, falling back to a cheap word-based
estimate so the app stays offline-capable even if the encoding can't be loaded.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

try:
    import tiktoken

    _ENC = tiktoken.get_encoding("cl100k_base")
except Exception:  # pragma: no cover - offline / download failure fallback
    _ENC = None


def count_tokens(text: str) -> int:
    if _ENC is not None:
        try:
            return len(_ENC.encode(text))
        except Exception:
            pass
    # ~4 chars per token heuristic.
    return max(1, len(text) // 4)


_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")


def split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    # Split on paragraph breaks first, then sentence boundaries within.
    pieces: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        for s in _SENTENCE_RE.split(para):
            s = s.strip()
            if s:
                pieces.append(s)
    return pieces


@dataclass
class ChunkResult:
    text: str
    token_count: int
    char_start: int
    char_end: int


def sentence_window_chunks(
    text: str, window_tokens: int = 512, overlap_tokens: int = 64
) -> list[ChunkResult]:
    """Accumulate sentences into windows of ~window_tokens, carrying roughly
    overlap_tokens of trailing sentences into the next window."""
    sentences = split_sentences(text)
    if not sentences:
        return []

    results: list[ChunkResult] = []
    cur: list[str] = []
    cur_tokens = 0

    def flush() -> None:
        nonlocal cur, cur_tokens
        if not cur:
            return
        joined = " ".join(cur)
        start = text.find(cur[0])
        if start < 0:
            start = 0
        end = start + len(joined)
        results.append(ChunkResult(joined, count_tokens(joined), start, end))

    for sent in sentences:
        st = count_tokens(sent)
        if cur and cur_tokens + st > window_tokens:
            flush()
            # Build overlap tail from the end of the current window.
            tail: list[str] = []
            tail_tokens = 0
            for s in reversed(cur):
                t = count_tokens(s)
                if tail_tokens + t > overlap_tokens:
                    break
                tail.insert(0, s)
                tail_tokens += t
            cur = tail
            cur_tokens = tail_tokens
        cur.append(sent)
        cur_tokens += st

    flush()
    return results
