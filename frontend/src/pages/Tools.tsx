import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../api";
import type { Project, ScrapeJob, ScreenScraperStatus } from "../types";
import { Badge, useToast } from "../ui";

type Point = { x: number; y: number };
type Region = { left: number; top: number; width: number; height: number };

const ACTIVE = ["queued", "running"];

export default function Tools({ project }: { project: Project }) {
  const toast = useToast();
  const [status, setStatus] = useState<ScreenScraperStatus | null>(null);
  const [shotUrl, setShotUrl] = useState<string | null>(null);
  const [mode, setMode] = useState<"click" | "region">("region");
  const [click, setClick] = useState<Point | null>(null);
  const [region, setRegion] = useState<Region | null>(null);
  const [dragStart, setDragStart] = useState<Point | null>(null);

  const [title, setTitle] = useState("");
  const [pause, setPause] = useState(0.2);
  const [maxPages, setMaxPages] = useState(500);

  const [activeJob, setActiveJob] = useState<ScrapeJob | null>(null);
  const [jobs, setJobs] = useState<ScrapeJob[]>([]);
  const [busy, setBusy] = useState(false);
  const imgRef = useRef<HTMLImageElement>(null);

  const screenW = status?.screen?.physical_width ?? 0;
  const screenH = status?.screen?.physical_height ?? 0;

  const loadStatus = useCallback(() => api.scraperStatus().then(setStatus).catch(() => {}), []);
  const loadJobs = useCallback(
    () => api.listScrapeJobs(project.id).then(setJobs).catch(() => {}),
    [project.id],
  );

  useEffect(() => { loadStatus(); }, [loadStatus]);
  useEffect(() => { loadJobs(); }, [loadJobs]);

  // Poll the active job while it runs.
  useEffect(() => {
    if (!activeJob || !ACTIVE.includes(activeJob.status)) return;
    const t = window.setInterval(async () => {
      try {
        const j = await api.getScrapeJob(activeJob.id);
        setActiveJob(j);
        if (!ACTIVE.includes(j.status)) {
          loadJobs();
          if (j.status === "done") {
            toast(j.source_id
              ? `Capture complete — ${j.pages} page(s) saved as a source.`
              : "Capture finished, but no text was recognised.");
          } else if (j.status === "error") {
            toast(j.error || "Scrape failed.", true);
          }
        }
      } catch { /* ignore poll errors */ }
    }, 1000);
    return () => window.clearInterval(t);
  }, [activeJob, loadJobs, toast]);

  const capture = async () => {
    setBusy(true);
    try {
      // Fetch the bytes first so a failed capture surfaces the server's reason
      // instead of leaving an empty box behind a broken <img>.
      const blob = await api.scraperScreenshot();
      setShotUrl((prev) => {
        if (prev?.startsWith("blob:")) URL.revokeObjectURL(prev);
        return URL.createObjectURL(blob);
      });
      setClick(null);
      setRegion(null);
      loadStatus();
    } catch (e) {
      toast((e as Error).message || "Could not capture the screen.", true);
    } finally {
      setBusy(false);
    }
  };

  // Release the last object URL when the component unmounts.
  useEffect(() => () => {
    setShotUrl((prev) => { if (prev?.startsWith("blob:")) URL.revokeObjectURL(prev); return prev; });
  }, []);

  // Map a mouse event to physical screenshot pixels.
  const toPhysical = (e: React.MouseEvent): Point => {
    const img = imgRef.current!;
    const rect = img.getBoundingClientRect();
    const nx = img.naturalWidth || screenW || rect.width;
    const ny = img.naturalHeight || screenH || rect.height;
    const x = Math.min(nx, Math.max(0, ((e.clientX - rect.left) / rect.width) * nx));
    const y = Math.min(ny, Math.max(0, ((e.clientY - rect.top) / rect.height) * ny));
    return { x: Math.round(x), y: Math.round(y) };
  };

  const onDown = (e: React.MouseEvent) => {
    const p = toPhysical(e);
    if (mode === "click") {
      setClick(p);
    } else {
      setDragStart(p);
      setRegion({ left: p.x, top: p.y, width: 0, height: 0 });
    }
  };
  const onMove = (e: React.MouseEvent) => {
    if (mode !== "region" || !dragStart) return;
    const p = toPhysical(e);
    setRegion({
      left: Math.min(dragStart.x, p.x),
      top: Math.min(dragStart.y, p.y),
      width: Math.abs(p.x - dragStart.x),
      height: Math.abs(p.y - dragStart.y),
    });
  };
  const onUp = () => setDragStart(null);

  const pct = (v: number, total: number) => `${total ? (v / total) * 100 : 0}%`;

  const canRun = status?.available && click && region && region.width > 4 && region.height > 4;

  const run = async () => {
    if (!canRun || !click || !region) return;
    setBusy(true);
    try {
      const job = await api.startScrape({
        project_id: project.id,
        title: title.trim(),
        region_left: region.left,
        region_top: region.top,
        region_width: region.width,
        region_height: region.height,
        click_x: click.x,
        click_y: click.y,
        pause_seconds: pause,
        max_pages: maxPages,
        change_threshold: 0.01,
      });
      setActiveJob(job);
      loadJobs();
      toast("Scrape started — switch to the target app; do not move its window.");
    } catch (e) {
      toast((e as Error).message, true);
    } finally {
      setBusy(false);
    }
  };

  const cancel = async () => {
    if (!activeJob) return;
    await api.cancelScrape(activeJob.id);
    toast("Cancelling…");
  };

  const running = activeJob && ACTIVE.includes(activeJob.status);

  return (
    <div>
      <h1 className="page-title">Tools</h1>
      <p className="page-sub">Utilities that feed text into this project for LLM training.</p>

      <div className="card" style={{ marginBottom: 18 }}>
        <h3>🖱️ Screen Text Scraper</h3>
        <p className="muted" style={{ marginTop: -6, fontSize: 13 }}>
          OCRs a region of your screen, sends a mouse click to advance the other app
          (e.g. turn a page), and repeats until the region stops changing. The combined
          text is saved as a source in <strong>{project.name}</strong>.
        </p>

        {/* Availability banner */}
        {status && !status.available && (
          <div className="badge red" style={{ display: "block", padding: 12, marginBottom: 12, lineHeight: 1.5 }}>
            Tool unavailable: {status.detail}
            {status.missing.length > 0 && <> Missing: {status.missing.join(", ")}.</>}
          </div>
        )}

        {/* Step 1 — capture the screen */}
        <div className="row" style={{ gap: 10, marginBottom: 12, flexWrap: "wrap" }}>
          <button className="btn" onClick={capture} disabled={busy || (status ? !status.available : true)}>
            {busy ? "Capturing…" : shotUrl ? "Recapture screen" : "Capture screen"}
          </button>
          {shotUrl && (
            <>
              <button
                className={`btn small ${mode === "region" ? "" : "ghost"}`}
                onClick={() => setMode("region")}
              >
                ▭ Highlight capture area
              </button>
              <button
                className={`btn small ${mode === "click" ? "" : "ghost"}`}
                onClick={() => setMode("click")}
              >
                ＋ Set click point
              </button>
            </>
          )}
        </div>

        {/* Step 2 — the picker */}
        {shotUrl && (
          <div className="scrape-picker" onMouseLeave={onUp}>
            <img
              ref={imgRef}
              src={shotUrl}
              alt="screen"
              draggable={false}
              onMouseDown={onDown}
              onMouseMove={onMove}
              onMouseUp={onUp}
              onError={() => toast("The captured image could not be displayed.", true)}
              style={{ cursor: mode === "click" ? "crosshair" : "cell" }}
            />
            {region && (
              <div
                className="scrape-region"
                style={{
                  left: pct(region.left, screenW),
                  top: pct(region.top, screenH),
                  width: pct(region.width, screenW),
                  height: pct(region.height, screenH),
                }}
              />
            )}
            {click && (
              <div
                className="scrape-click"
                style={{ left: pct(click.x, screenW), top: pct(click.y, screenH) }}
                title={`click @ ${click.x}, ${click.y}`}
              />
            )}
          </div>
        )}

        {shotUrl && (
          <div className="muted" style={{ fontSize: 12, margin: "8px 0 4px" }}>
            {mode === "region" ? "Drag a rectangle over the area the text appears in." : "Click where the advance/next click should land."}
            {region && ` · area ${region.width}×${region.height}px`}
            {click && ` · click @ ${click.x},${click.y}`}
          </div>
        )}

        {/* Step 3 — options + run */}
        {shotUrl && (
          <div className="grid cols-2" style={{ gap: 12, marginTop: 12 }}>
            <label className="field">
              <span>Source title (optional)</span>
              <input value={title} onChange={(e) => setTitle(e.target.value)} placeholder="e.g. Scanned manual" />
            </label>
            <div className="row" style={{ gap: 12 }}>
              <label className="field" style={{ flex: 1 }}>
                <span>Pause before click (s)</span>
                <input type="number" step="0.1" min="0" value={pause}
                  onChange={(e) => setPause(Number(e.target.value))} />
              </label>
              <label className="field" style={{ flex: 1 }}>
                <span>Max pages</span>
                <input type="number" min="1" value={maxPages}
                  onChange={(e) => setMaxPages(Number(e.target.value))} />
              </label>
            </div>
          </div>
        )}

        {shotUrl && (
          <div className="row" style={{ gap: 10, marginTop: 14 }}>
            {!running ? (
              <button className="btn" onClick={run} disabled={!canRun || busy}>
                ▶ Run scrape
              </button>
            ) : (
              <button className="btn danger" onClick={cancel}>■ Stop</button>
            )}
            {!canRun && !running && (
              <span className="muted" style={{ fontSize: 12, alignSelf: "center" }}>
                Set both a capture area and a click point to run.
              </span>
            )}
          </div>
        )}

        {/* Live progress */}
        {activeJob && (
          <div className="card" style={{ marginTop: 14, background: "var(--bg)" }}>
            <div className="row" style={{ justifyContent: "space-between" }}>
              <strong>{running ? "Scraping…" : "Last run"}</strong>
              <Badge status={activeJob.status} />
            </div>
            <div className="muted" style={{ fontSize: 12, marginTop: 6 }}>
              {activeJob.pages} page(s) captured · {(activeJob.text || "").length.toLocaleString()} chars
              {activeJob.source_id && " · saved as a source"}
            </div>
            {activeJob.text && (
              <pre style={{
                marginTop: 8, maxHeight: 180, overflow: "auto", fontSize: 12,
                whiteSpace: "pre-wrap", color: "var(--muted)",
              }}>
                {activeJob.text.slice(-2000)}
              </pre>
            )}
          </div>
        )}
      </div>

      {/* History */}
      <div className="card">
        <h3>Recent scrapes ({jobs.length})</h3>
        {jobs.length === 0 ? (
          <div className="muted">No scrapes yet.</div>
        ) : (
          <table>
            <thead>
              <tr><th>Title</th><th>Status</th><th>Pages</th><th>When</th></tr>
            </thead>
            <tbody>
              {jobs.map((j) => (
                <tr key={j.id}>
                  <td>{j.title || <span className="muted">(untitled)</span>}</td>
                  <td><Badge status={j.status} /></td>
                  <td>{j.pages}</td>
                  <td className="muted" style={{ fontSize: 12 }}>
                    {new Date(j.created_at).toLocaleString()}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
