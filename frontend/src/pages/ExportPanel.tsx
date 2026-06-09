import { useState } from "react";
import { api } from "../api";
import type { ExportResult, Meta, Project, Stats } from "../types";
import { useToast } from "../ui";

interface Props {
  project: Project;
  meta: Meta;
  stats: Stats | null;
}

const STATUSES = ["approved", "pending_review", "edited", "rejected"];

export default function ExportPanel({ project, meta, stats }: Props) {
  const [format, setFormat] = useState(project.config.export.format);
  const [split, setSplit] = useState(project.config.export.train_split);
  const [statuses, setStatuses] = useState<string[]>(project.config.export.include_statuses);
  const [result, setResult] = useState<ExportResult | null>(null);
  const [busy, setBusy] = useState(false);
  const toast = useToast();

  const toggle = (s: string) =>
    setStatuses((prev) => (prev.includes(s) ? prev.filter((x) => x !== s) : [...prev, s]));

  const selectable = stats
    ? statuses.reduce((acc, s) => acc + (stats.by_status[s] || 0), 0)
    : 0;

  const doExport = async () => {
    setBusy(true);
    try {
      const r = await api.exportDataset(project.id, {
        format, train_split: split, include_statuses: statuses,
      });
      setResult(r);
      toast(`Exported ${r.train_count + r.val_count} samples.`);
    } catch (e) {
      toast((e as Error).message, true);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div>
      <h1 className="page-title">Export Dataset</h1>
      <p className="page-sub">Write train/val JSONL files plus a reproducibility manifest to the project folder.</p>

      <div className="grid cols-2">
        <div>
          <div className="card" style={{ marginBottom: 14 }}>
            <h3>Format</h3>
            <select value={format} onChange={(e) => setFormat(e.target.value)}>
              {meta.export_formats.map((f) => <option key={f} value={f}>{f}</option>)}
            </select>
            <p className="muted" style={{ fontSize: 12, marginTop: 10 }}>
              {format === "alpaca" && "{ instruction, input, output } JSONL"}
              {format === "sharegpt" && "{ conversations: [human, gpt] } JSONL"}
              {format === "openai" && "{ messages: [user, assistant] } JSONL"}
            </p>
          </div>

          <div className="card" style={{ marginBottom: 14 }}>
            <h3>Train / validation split</h3>
            <input type="range" min={0.5} max={1} step={0.05} value={split}
              onChange={(e) => setSplit(+e.target.value)} style={{ width: "100%" }} />
            <div className="muted">{Math.round(split * 100)}% train / {Math.round((1 - split) * 100)}% val · stratified by type</div>
          </div>

          <div className="card" style={{ marginBottom: 14 }}>
            <h3>Include statuses</h3>
            <div className="pill-row">
              {STATUSES.map((s) => (
                <label key={s} className="checkbox">
                  <input type="checkbox" checked={statuses.includes(s)} onChange={() => toggle(s)} />
                  {s.replace(/_/g, " ")} <span className="muted">({stats?.by_status[s] || 0})</span>
                </label>
              ))}
            </div>
          </div>

          <button className="btn" onClick={doExport} disabled={busy || selectable === 0}>
            {busy ? "Exporting…" : `Export ${selectable} samples`}
          </button>

          {result && (
            <div className="card" style={{ marginTop: 14 }}>
              <h3>Export complete</h3>
              <div>Train: <strong>{result.train_count}</strong> · Validation: <strong>{result.val_count}</strong></div>
              <div className="pill-row" style={{ marginTop: 10 }}>
                <a className="btn small secondary" href={api.downloadUrl(project.id, result.train_file)}>train.jsonl</a>
                <a className="btn small secondary" href={api.downloadUrl(project.id, result.val_file)}>val.jsonl</a>
                <a className="btn small ghost" href={api.downloadUrl(project.id, result.manifest_file)}>manifest.json</a>
              </div>
              <div className="mono muted" style={{ fontSize: 11, marginTop: 10 }}>{result.train_file}</div>
            </div>
          )}
        </div>

        <div className="card">
          <h3>Dataset statistics</h3>
          {!stats || stats.samples === 0 ? (
            <div className="muted">No samples yet.</div>
          ) : (
            <>
              <div className="muted" style={{ fontSize: 12, marginBottom: 4 }}>By type</div>
              {Object.entries(stats.by_type).map(([k, v]) => (
                <Bar key={k} label={k} value={v} max={stats.samples} />
              ))}
              <div className="muted" style={{ fontSize: 12, margin: "14px 0 4px" }}>By status</div>
              {Object.entries(stats.by_status).map(([k, v]) => (
                <Bar key={k} label={k.replace(/_/g, " ")} value={v} max={stats.samples} />
              ))}
              {Object.keys(stats.avg_quality).length > 0 && (
                <>
                  <div className="muted" style={{ fontSize: 12, margin: "14px 0 4px" }}>Avg quality</div>
                  {Object.entries(stats.avg_quality).map(([k, v]) => (
                    <Bar key={k} label={k} value={v} max={5} suffix={`${v}`} />
                  ))}
                </>
              )}
              <div className="muted" style={{ fontSize: 12, margin: "14px 0 4px" }}>Chunk token distribution</div>
              {stats.token_histogram.map((h) => (
                <Bar key={h.bucket} label={h.bucket} value={h.count}
                  max={Math.max(...stats.token_histogram.map((x) => x.count), 1)} />
              ))}
            </>
          )}
        </div>
      </div>
    </div>
  );
}

function Bar({ label, value, max, suffix }: { label: string; value: number; max: number; suffix?: string }) {
  const pct = max > 0 ? (value / max) * 100 : 0;
  return (
    <div className="bar">
      <div className="bar-label">{label}</div>
      <div className="bar-track"><div className="bar-fill" style={{ width: `${pct}%` }} /></div>
      <div className="bar-num">{suffix ?? value}</div>
    </div>
  );
}
