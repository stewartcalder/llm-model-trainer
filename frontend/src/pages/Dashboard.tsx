import { useEffect, useState } from "react";
import { api } from "../api";
import type { Project, Run, Stats } from "../types";
import { Badge, fmtCost, fmtNum } from "../ui";

interface Props {
  project: Project;
  stats: Stats | null;
  go: (v: "sources" | "configure" | "run" | "review" | "export") => void;
}

function Stat({ label, value, sub }: { label: string; value: string | number; sub?: string }) {
  return (
    <div className="card stat">
      <div className="label">{label}</div>
      <div className="value">{value}</div>
      {sub && <div className="sub">{sub}</div>}
    </div>
  );
}

export default function Dashboard({ project, stats, go }: Props) {
  const [runs, setRuns] = useState<Run[]>([]);

  useEffect(() => {
    api.listRuns(project.id).then(setRuns);
  }, [project.id]);

  return (
    <div>
      <h1 className="page-title">{project.name}</h1>
      <p className="page-sub">
        Turn curated sources into a fine-tuning dataset — ingest, generate, review, export.
      </p>

      <div className="grid cols-4" style={{ marginBottom: 16 }}>
        <Stat label="Sources" value={stats?.sources ?? 0} />
        <Stat label="Chunks" value={stats?.chunks ?? 0} />
        <Stat label="Samples" value={stats?.samples ?? 0} />
        <Stat
          label="Approved"
          value={`${stats?.approved_pct ?? 0}%`}
          sub={`${stats?.by_status.approved ?? 0} ready to export`}
        />
      </div>

      <div className="card" style={{ marginBottom: 16 }}>
        <h3>Quick actions</h3>
        <div className="row wrap">
          <button className="btn" onClick={() => go("sources")}>Add sources</button>
          <button className="btn secondary" onClick={() => go("configure")}>Configure pipeline</button>
          <button className="btn secondary" onClick={() => go("run")}>Run pipeline</button>
          <button className="btn secondary" onClick={() => go("review")}>Review samples</button>
          <button className="btn secondary" onClick={() => go("export")}>Export dataset</button>
        </div>
      </div>

      <div className="card">
        <h3>Recent runs</h3>
        {runs.length === 0 ? (
          <div className="muted">No runs yet. Configure the pipeline and start a run.</div>
        ) : (
          <table>
            <thead>
              <tr>
                <th>Started</th>
                <th>Status</th>
                <th>Chunks</th>
                <th>Samples</th>
                <th>Tokens</th>
                <th>Cost</th>
              </tr>
            </thead>
            <tbody>
              {runs.map((r) => (
                <tr key={r.id}>
                  <td>{new Date(r.started_at).toLocaleString()}</td>
                  <td><Badge status={r.status} /></td>
                  <td>{r.chunks_processed}/{r.chunks_total}</td>
                  <td>{r.samples_generated}</td>
                  <td>{fmtNum(r.tokens_used)}</td>
                  <td>{fmtCost(r.cost_usd)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
