"""Source ingestion: PDF text extraction and URL article fetching (spec 5.1)."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field


@dataclass
class ExtractedDoc:
    """Extracted text, split into labelled sections (e.g. PDF pages)."""

    title: str
    sections: list[tuple[str, str]] = field(default_factory=list)  # (label, text)

    @property
    def full_text(self) -> str:
        return "\n\n".join(t for _, t in self.sections)

    def content_hash(self) -> str:
        return hashlib.sha256(self.full_text.encode("utf-8", "ignore")).hexdigest()


def extract_pdf(path: str) -> ExtractedDoc:
    from pypdf import PdfReader

    reader = PdfReader(path)
    sections: list[tuple[str, str]] = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        text = text.strip()
        if text:
            sections.append((f"page {i}", text))

    meta_title = ""
    try:
        if reader.metadata and reader.metadata.title:
            meta_title = str(reader.metadata.title)
    except Exception:
        meta_title = ""

    return ExtractedDoc(title=meta_title, sections=sections)


def extract_url(url: str) -> ExtractedDoc:
    import httpx
    import trafilatura

    headers = {"User-Agent": "Mozilla/5.0 (LoRA-Data-Builder/0.1)"}
    with httpx.Client(follow_redirects=True, timeout=30.0, headers=headers) as client:
        resp = client.get(url)
        resp.raise_for_status()
        html = resp.text

    text = trafilatura.extract(html, include_comments=False, include_tables=True) or ""
    title = ""
    meta = trafilatura.extract_metadata(html)
    if meta and meta.title:
        title = meta.title
    if not text.strip():
        raise ValueError("No readable article text could be extracted from the URL.")

    return ExtractedDoc(title=title, sections=[("article", text.strip())])
