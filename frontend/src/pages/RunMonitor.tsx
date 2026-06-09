import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../api";
import type { DryRun, Project, Run, Sample } from "../types";
import { Badge, fmtCost, fmtNum, useToast } from "../ui";

const STAGES = ["ingesting", "chunking", "generating", "done"];
const ACTIVE = ["queued", "running"];

interface Props {
  project: Project;
  onChanged: () => void;
  goReview: () => void;
}

export default function RunMonitor({ project, onChanged, goReview }: Props) {
  const [dry, setDry] = useState<DryRun | null>(null);
  const [run, setRun] = useState<Run | null>(null);
  const [preview, setPreview] = useState<Sample[]>([]);
  const [starting, setStarting] = useState(false);
  const timer = useRef<number | null>(null);
  const toast = useToast();

  const loadDry = useCallback(() => {
    api.dryRun(project.id).then(setDry).catch(() => {});
  }, [project.id]);

  const loadLatest = useCallback(async () => {
    const runs = await api.listRuns(project.id);
    if (runs.length) setRun(runs[0]);
  }, [project.id]);

  useEffect(() => {
    loadDry();
    loadLatest();
  }, [loadDry, loadLatest]);

  // Poll while a run is active.
  useEffect(() => {
    const active = run && ACTIVE.includes(run.status);
    if (active) {
      timer.current = window.setInterval(async () => {
        const r = await api.getRun(project.id, run!.id);
        setRun(r);
        const p = await api.listSamples(project.id, { limit: "5" });
        setPreview(p);
        onChanged();
        if (!ACTIVE.includes(r.status)) {
          loadDry();
        }
      }, 1000);
    }
    return () => {
      if (timer.current) window.clearInterval(timer.current);
    };
  }, [run, project.id, onChanged, loadDry]);

  const start = async () => {
    setStarting(true);
    try {
      const r = await api.startRun(project.id);
      setRun(r);
      toast("Run started.");
    } catch (e) {
      toast((e as Error).message, true);
    } finally {
      setStarting(false);
    }
  };

  const cancel = async () => {
    if (!run) return;
    await api.cancelRun(project.id, run.id);
    toast("Cancellation requested.");
  };

  const isActive = run && ACTIVE.includes(run.status);
  const pct = run && run.chunks_total > 0
    ? Math.round((run.chunks_processed / run.chunks_total) * 100)
    : 0;
  const stageIdx = run ? STAGES.indexOf(run.stage) : -1;

  return (
    <div>
      <h1 className="page-title">Run Pipeline</h1>
      <p className="page-sub">Ingest sources, chunk them, and generate samples with the configured LLM.</p>

      <div className="card" style={{ marginBottom: 14 }}>
        <h3>Dry-run estimate</h3>
        {dry ? (
          <div className="grid cols-4">
            <div className="stat"><div className="label">Pending sources</div><div className="value">{dry.pending_sources}</div></div>
            <div className="stat"><div className="label">Est. chunks</div><div className="value">{dry.estimated_chunks}</div></div>
            <div className="stat"><div className="label">Est. LLM calls</div><div className="value">{fmtNum(dry.estimated_calls)}</div></div>
            <div className="stat"><div className="label">Est. cost</div><div className="value">{fmtCost(dry.estimated_cost_usd)}</div><div className="sub">{fmtNum(dry.estimated_tokens)} tokens</div></div>
          </div>
        ) : <div className="muted">Estimating…</div>}
        <div className="row" style={{ marginTop: 16 }}>
          <button className="btn" onClick={start} disabled={!!isActive || starting}>
            {isActive ? "Running…" : starting ? "Starting…" : "▶ Start run"}
          </button>
          {isActive && <button className="btn danger" onClick={cancel}>Cancel</button>}
          <button className="btn ghost" onClick={loadDry}>Refresh estimate</button>
        </div>
      </div>

      {run && (
        <>
          <div className="card" style={{ marginBottom: 14 }}>
            <div className="row" style={{ marginBottom: 12 }}>
              <h3 style={{ margin: 0 }}>Progress</h3>
              <Badge status={run.status} />
              <div className="spacer" />
              <span className="muted">
                {fmtNum(run.tokens_used)} tokens · {fmtCost(run.cost_usd)} · {run.samples_generated} samples
              </span>
            </div>

            <div className="pill-row" style={{ marginBottom: 12 }}>
              {STAGES.map((s, i) => (
                <span key={s} className={`badge ${i <= stageIdx ? "blue" : ""}`}>
                  {i < stageIdx || run.status === "done" ? "✓ " : i === stageIdx && isActive ? "● " : ""}{s}
                </span>
              ))}
            </div>

            <div className="progress"><div style={{ width: `${pct}%` }} /></div>
            <div className="muted" style={{ fontSize: 12, marginTop: 6 }}>
              {run.chunks_processed} / {run.chunks_total} chunks processed
            </div>

            {run.status === "done" && run.samples_generated > 0 && (
              <button className="btn" style={{ marginTop: 14 }} onClick={goReview}>Review generated samples →</button>
            )}
          </div>

          <div className="grid cols-2">
            <div className="card">
              <h3>Live log</h3>
              <div className="log">{run.log || "Waiting…"}</div>
            </div>
            <div className="card">
              <h3>Latest samples</h3>
              {preview.length === 0 ? (
                <div className="muted">No samples yet.</div>
              ) : preview.map((s) => (
                <div key={s.id} style={{ marginBottom: 12, paddingBottom: 12, borderBottom: "1px solid var(--border)" }}>
                  <span className="badge blue">{s.type}</span> <Badge status={s.status} />
                  <div style={{ marginTop: 6, fontWeight: 600 }}>{s.instruction}</div>
                  <div className="muted" style={{ marginTop: 4 }}>{s.output.slice(0, 180)}{s.output.length > 180 ? "…" : ""}</div>
                </div>
              ))}
            </div>
          </div>
        </>
      )}
    </div>
  );
}
