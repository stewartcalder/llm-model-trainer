"""Source ingestion: text extraction for PDF, DOCX, TXT, MD, and URLs (spec 5.1)."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ExtractedDoc:
    """Extracted text split into labelled sections (e.g. PDF pages, headings)."""

    title: str
    sections: list[tuple[str, str]] = field(default_factory=list)  # (label, text)

    @property
    def full_text(self) -> str:
        return "\n\n".join(t for _, t in self.sections)

    def content_hash(self) -> str:
        return hashlib.sha256(self.full_text.encode("utf-8", "ignore")).hexdigest()


# ---- File extractors ----

def extract_pdf(path: str) -> ExtractedDoc:
    from pypdf import PdfReader

    reader = PdfReader(path)
    sections: list[tuple[str, str]] = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        if text.strip():
            sections.append((f"page {i}", text.strip()))

    title = ""
    try:
        if reader.metadata and reader.metadata.title:
            title = str(reader.metadata.title)
    except Exception:
        pass

    return ExtractedDoc(title=title, sections=sections)


def extract_docx(path: str) -> ExtractedDoc:
    from docx import Document

    doc = Document(path)
    title = ""
    paragraphs: list[str] = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        # Use the first heading as the document title.
        if not title and para.style.name.startswith("Heading"):
            title = text
        paragraphs.append(text)

    full = "\n\n".join(paragraphs)
    return ExtractedDoc(title=title, sections=[("document", full)])


def extract_text(path: str) -> ExtractedDoc:
    """Plain text and Markdown files — treat as a single section."""
    text = Path(path).read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        raise ValueError("File appears to be empty.")
    # Use the first non-empty line as a title candidate.
    first_line = next((l.lstrip("#").strip() for l in text.splitlines() if l.strip()), "")
    return ExtractedDoc(title=first_line[:120], sections=[("document", text)])


# ---- URL extractor ----

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


# ---- Dispatch ----

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".markdown"}


def extract_file(path: str) -> ExtractedDoc:
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return extract_pdf(path)
    if ext == ".docx":
        return extract_docx(path)
    if ext in (".txt", ".md", ".markdown"):
        return extract_text(path)
    raise ValueError(f"Unsupported file type: {ext}")
