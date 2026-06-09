import { Fragment, useCallback, useEffect, useState } from "react";
import { api } from "../api";
import type { Project, Sample, Stats } from "../types";
import { Badge, useToast } from "../ui";

interface Props {
  project: Project;
  stats: Stats | null;
  onChanged: () => void;
}

function QualityChips({ q }: { q: Sample["quality"] }) {
  const dims = ["faithfulness", "completeness", "clarity"] as const;
  const present = dims.filter((d) => typeof q[d] === "number");
  if (present.length === 0) return null;
  return (
    <span className="pill-row" style={{ display: "inline-flex", marginLeft: 8 }}>
      {present.map((d) => {
        const v = q[d] as number;
        return <span key={d} className={`badge ${v < 3 ? "red" : "green"}`}>{d[0].toUpperCase()}:{v}</span>;
      })}
    </span>
  );
}

export default function Review({ project, stats, onChanged }: Props) {
  const [samples, setSamples] = useState<Sample[]>([]);
  const [filters, setFilters] = useState({ status: "", type: "", search: "" });
  const [expanded, setExpanded] = useState<string | null>(null);
  const [edit, setEdit] = useState<Partial<Sample>>({});
  const toast = useToast();

  const load = useCallback(async () => {
    const params: Record<string, string> = {};
    if (filters.status) params.status = filters.status;
    if (filters.type) params.type = filters.type;
    if (filters.search) params.search = filters.search;
    setSamples(await api.listSamples(project.id, params));
  }, [project.id, filters]);

  useEffect(() => { load(); }, [load]);

  const setStatus = async (s: Sample, status: string) => {
    await api.updateSample(project.id, s.id, { status });
    await load();
    onChanged();
  };

  const startEdit = (s: Sample) => {
    setExpanded(s.id);
    setEdit({ instruction: s.instruction, input: s.input, output: s.output });
  };

  const saveEdit = async (s: Sample) => {
    await api.updateSample(project.id, s.id, edit);
    setExpanded(null);
    await load();
    onChanged();
    toast("Sample updated.");
  };

  const bulk = async (action: string) => {
    const { changed } = await api.bulk(project.id, action);
    await load();
    onChanged();
    toast(`${changed} sample(s) updated.`);
  };

  return (
    <div>
      <h1 className="page-title">Dataset Review</h1>
      <p className="page-sub">Approve, reject, or edit samples. Critic scores below 3 flag a sample for review.</p>

      <div className="card" style={{ marginBottom: 14 }}>
        <div className="row wrap">
          <select value={filters.status} onChange={(e) => setFilters({ ...filters, status: e.target.value })} style={{ width: 180 }}>
            <option value="">All statuses</option>
            {["pending_review", "approved", "rejected", "edited"].map((s) => (
              <option key={s} value={s}>{s.replace(/_/g, " ")}{stats?.by_status[s] ? ` (${stats.by_status[s]})` : ""}</option>
            ))}
          </select>
          <select value={filters.type} onChange={(e) => setFilters({ ...filters, type: e.target.value })} style={{ width: 150 }}>
            <option value="">All types</option>
            <option value="qa">Q&A</option>
            <option value="instruction">Instruction</option>
          </select>
          <input type="text" placeholder="Search text…" style={{ width: 240 }}
            value={filters.search} onChange={(e) => setFilters({ ...filters, search: e.target.value })} />
          <div className="spacer" />
          <button className="btn small secondary" onClick={() => bulk("approve_all")}>Approve all</button>
          <button className="btn small secondary" onClick={() => bulk("reject_flagged")}>Reject flagged</button>
        </div>
      </div>

      <div className="card">
        {samples.length === 0 ? (
          <div className="empty">No samples match these filters.</div>
        ) : (
          <table>
            <thead>
              <tr>
                <th style={{ width: 110 }}>Type</th>
                <th>Instruction</th>
                <th style={{ width: 110 }}>Status</th>
                <th style={{ width: 220 }}>Actions</th>
              </tr>
            </thead>
            <tbody>
              {samples.map((s) => (
                <Fragment key={s.id}>
                  <tr>
                    <td><span className="badge blue">{s.type}</span></td>
                    <td>
                      <div onClick={() => setExpanded(expanded === s.id ? null : s.id)} style={{ cursor: "pointer" }}>
                        {s.instruction}
                        <QualityChips q={s.quality} />
                      </div>
                      <div className="muted" style={{ fontSize: 11, marginTop: 2 }}>from {s.source_title}</div>
                    </td>
                    <td><Badge status={s.status} /></td>
                    <td>
                      <div className="row">
                        <button className="btn small secondary" onClick={() => setStatus(s, "approved")}>✓</button>
                        <button className="btn small danger" onClick={() => setStatus(s, "rejected")}>✕</button>
                        <button className="btn small ghost" onClick={() => startEdit(s)}>Edit</button>
                      </div>
                    </td>
                  </tr>
                  {expanded === s.id && (
                    <tr>
                      <td colSpan={4} style={{ background: "var(--bg)" }}>
                        <div className="sample-fields">
                          <label className="field">
                            <span>Instruction</span>
                            <textarea value={edit.instruction ?? s.instruction}
                              onChange={(e) => setEdit({ ...edit, instruction: e.target.value })} />
                          </label>
                          {(s.type === "instruction") && (
                            <label className="field">
                              <span>Input</span>
                              <textarea value={edit.input ?? s.input}
                                onChange={(e) => setEdit({ ...edit, input: e.target.value })} />
                            </label>
                          )}
                          <label className="field">
                            <span>Output</span>
                            <textarea value={edit.output ?? s.output} rows={4}
                              onChange={(e) => setEdit({ ...edit, output: e.target.value })} />
                          </label>
                          <details>
                            <summary className="muted" style={{ cursor: "pointer" }}>Source chunk & quality</summary>
                            <div className="muted" style={{ fontSize: 12, marginTop: 6 }}>{s.chunk_text}</div>
                            {s.quality.reason && <div className="muted" style={{ marginTop: 6, fontSize: 12 }}>Critic: {String(s.quality.reason)}</div>}
                          </details>
                          <div className="row">
                            <button className="btn small" onClick={() => saveEdit(s)}>Save changes</button>
                            <button className="btn small ghost" onClick={() => setExpanded(null)}>Close</button>
                          </div>
                        </div>
                      </td>
                    </tr>
                  )}
                </Fragment>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
