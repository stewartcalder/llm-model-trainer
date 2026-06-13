"""Screen-text-scraper tool.

Drives a click-through OCR loop against another application running on the same
machine as the backend:

    1. grab the configured screen region,
    2. OCR it and append the text,
    3. pause briefly, then send a mouse click,
    4. grab the region again and compare it to the previous frame,
    5. if it changed, repeat; if it did not, the document has reached its end.

The accumulated text is written out as a `Source` (type "screen") so it feeds the
normal ingest -> chunk -> sample pipeline used for LLM training.

All screen/mouse libraries (mss, pyautogui, pytesseract, Pillow) are imported
lazily so the rest of the app runs on headless / server installs where these are
absent. `availability()` reports what is missing.

Coordinate spaces: the frontend works entirely in *physical* screenshot pixels
(what `mss` captures). `pyautogui` clicks in *logical* pixels, so click coords are
scaled by ``logical_width / physical_width`` before the click is sent.
"""
from __future__ import annotations

import asyncio
import hashlib
import io
from datetime import datetime, timezone

from sqlalchemy import select

from . import models
from .config import UPLOAD_DIR
from .database import SessionLocal

# Job id -> asyncio.Event used to request cancellation.
CANCEL_FLAGS: dict[str, asyncio.Event] = {}


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Capability probing
# ---------------------------------------------------------------------------

def _probe_imports() -> list[str]:
    """Return the list of required modules that cannot be imported."""
    missing: list[str] = []
    for mod, label in (("mss", "mss"), ("PIL", "Pillow"),
                       ("pytesseract", "pytesseract"), ("pyautogui", "pyautogui")):
        try:
            __import__(mod)
        except Exception:  # noqa: BLE001 — any import error means unavailable
            missing.append(label)
    return missing


def _screen_dims() -> dict[str, int] | None:
    """Physical (capture) and logical (click) primary-monitor dimensions."""
    try:
        import mss  # type: ignore

        with mss.mss() as sct:
            mon = sct.monitors[1]  # 1 = primary monitor; 0 = the virtual union
            physical_w, physical_h = int(mon["width"]), int(mon["height"])
        logical_w, logical_h = physical_w, physical_h
        try:
            import pyautogui  # type: ignore

            size = pyautogui.size()
            logical_w, logical_h = int(size[0]), int(size[1])
        except Exception:  # noqa: BLE001 — fall back to physical == logical
            pass
        return {
            "physical_width": physical_w,
            "physical_height": physical_h,
            "logical_width": logical_w,
            "logical_height": logical_h,
            "mon_left": int(mon["left"]),
            "mon_top": int(mon["top"]),
        }
    except Exception:  # noqa: BLE001
        return None


def availability() -> dict:
    """Report whether the tool can run here, and the screen geometry if so."""
    missing = _probe_imports()
    if missing:
        return {
            "available": False,
            "missing": missing,
            "detail": "Install missing packages on the machine running the backend: "
                      + ", ".join(missing),
            "screen": None,
        }

    # Imports work — verify the tesseract binary and a usable display.
    detail = ""
    try:
        import pytesseract  # type: ignore

        pytesseract.get_tesseract_version()
    except Exception as exc:  # noqa: BLE001
        return {
            "available": False,
            "missing": ["tesseract-ocr (system binary)"],
            "detail": f"The tesseract OCR engine is not installed or not on PATH: {exc}",
            "screen": None,
        }

    screen = _screen_dims()
    if not screen:
        return {
            "available": False,
            "missing": ["display"],
            "detail": "No screen could be captured. The backend must run on a machine "
                      "with a graphical display (not headless).",
            "screen": None,
        }
    return {"available": True, "missing": [], "detail": detail, "screen": screen}


# ---------------------------------------------------------------------------
# Low-level capture / OCR / click primitives (blocking — call via to_thread)
# ---------------------------------------------------------------------------

def _grab_region(left: int, top: int, width: int, height: int):
    """Capture a screen region and return a PIL RGB Image (physical pixels)."""
    import mss  # type: ignore
    from PIL import Image  # type: ignore

    with mss.mss() as sct:
        shot = sct.grab({"left": left, "top": top, "width": width, "height": height})
    return Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")


def grab_full_screen_png() -> bytes:
    """Capture the whole primary monitor and return PNG bytes (for the picker UI)."""
    dims = _screen_dims()
    if not dims:
        raise RuntimeError("No capturable screen available.")
    img = _grab_region(dims["mon_left"], dims["mon_top"],
                       dims["physical_width"], dims["physical_height"])
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _ocr(image) -> str:
    import pytesseract  # type: ignore

    return pytesseract.image_to_string(image).strip()


def _click(physical_x: int, physical_y: int, dims: dict) -> None:
    """Send a mouse click, converting physical screenshot coords to logical ones."""
    import pyautogui  # type: ignore

    phys_w = max(1, dims.get("physical_width", 1))
    phys_h = max(1, dims.get("physical_height", 1))
    scale_x = dims.get("logical_width", phys_w) / phys_w
    scale_y = dims.get("logical_height", phys_h) / phys_h
    lx = dims.get("mon_left", 0) + int(round(physical_x * scale_x))
    ly = dims.get("mon_top", 0) + int(round(physical_y * scale_y))
    pyautogui.click(x=lx, y=ly)


def _change_ratio(img_a, img_b) -> float:
    """Fraction (0..1) of average per-pixel luminance difference between frames."""
    from PIL import ImageChops  # type: ignore

    if img_a is None or img_b is None or img_a.size != img_b.size:
        return 1.0
    diff = ImageChops.difference(img_a.convert("L"), img_b.convert("L"))
    hist = diff.histogram()  # 256 buckets of luminance-difference magnitude
    total = sum(hist)
    if not total:
        return 0.0
    weighted = sum(value * count for value, count in enumerate(hist))
    return weighted / (total * 255.0)


# ---------------------------------------------------------------------------
# Orchestration (async background task, mirrors pipeline.run_pipeline)
# ---------------------------------------------------------------------------

async def _persist(job_id: str, *, status: str | None = None, pages: int | None = None,
                   text: str | None = None, error: str | None = None,
                   source_id: str | None = None, finished: bool = False) -> None:
    async with SessionLocal() as db:
        job = await db.get(models.ScrapeJob, job_id)
        if not job:
            return
        if status is not None:
            job.status = status
        if pages is not None:
            job.pages = pages
        if text is not None:
            job.text = text
        if error is not None:
            job.error = error
        if source_id is not None:
            job.source_id = source_id
        if finished:
            job.finished_at = _now()
        await db.commit()


async def _store_source(job_id: str, project_id: str, title: str, text: str) -> str | None:
    """Write the captured text out as a pending Source and return its id."""
    if not text.strip():
        return None
    dest = UPLOAD_DIR / f"screen-{job_id}.txt"
    dest.write_text(text, encoding="utf-8")
    digest = hashlib.sha256(text.encode("utf-8", "ignore")).hexdigest()
    async with SessionLocal() as db:
        src = models.Source(
            project_id=project_id,
            type="screen",
            path_or_url=str(dest),
            title=title or f"Screen capture {_now():%Y-%m-%d %H:%M}",
            status="pending",
            content_hash=digest,
        )
        db.add(src)
        await db.commit()
        await db.refresh(src)
        return src.id


async def run_scrape_job(job_id: str, project_id: str, config: dict) -> None:
    cancel = CANCEL_FLAGS.setdefault(job_id, asyncio.Event())
    dims = _screen_dims()
    region = (
        int(config["region_left"]), int(config["region_top"]),
        int(config["region_width"]), int(config["region_height"]),
    )
    click_x, click_y = int(config["click_x"]), int(config["click_y"])
    pause = max(0.0, float(config.get("pause_seconds", 0.2)))
    max_pages = max(1, int(config.get("max_pages", 500)))
    threshold = float(config.get("change_threshold", 0.01))
    title = config.get("title", "")

    pieces: list[str] = []
    try:
        if not dims:
            raise RuntimeError("No capturable screen available on the backend host.")
        await _persist(job_id, status="running")

        prev = await asyncio.to_thread(_grab_region, *region)
        for page in range(max_pages):
            if cancel.is_set():
                break
            text = await asyncio.to_thread(_ocr, prev)
            if text:
                pieces.append(text)
            await _persist(job_id, pages=len(pieces), text="\n\n".join(pieces))

            await asyncio.sleep(pause)
            await asyncio.to_thread(_click, click_x, click_y, dims)

            current = await asyncio.to_thread(_grab_region, *region)
            if await asyncio.to_thread(_change_ratio, prev, current) <= threshold:
                break  # the region stopped changing — end of the document
            prev = current

        full_text = "\n\n".join(pieces)
        if cancel.is_set():
            await _persist(job_id, status="cancelled", text=full_text, finished=True)
            return

        source_id = await _store_source(job_id, project_id, title, full_text)
        await _persist(job_id, status="done", text=full_text,
                       source_id=source_id, finished=True)
    except Exception as exc:  # noqa: BLE001
        await _persist(job_id, status="error", text="\n\n".join(pieces),
                       error=str(exc), finished=True)
    finally:
        CANCEL_FLAGS.pop(job_id, None)


def request_cancel(job_id: str) -> bool:
    ev = CANCEL_FLAGS.get(job_id)
    if ev:
        ev.set()
        return True
    return False
