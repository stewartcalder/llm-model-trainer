import { useState } from "react";
import { api } from "../api";
import type { Project } from "../types";
import { useToast } from "../ui";

interface Props {
  projects: Project[];
  onSelect: (p: Project) => void;
  onChanged: () => void;
}

export default function Projects({ projects, onSelect, onChanged }: Props) {
  const [name, setName] = useState("");
  const [desc, setDesc] = useState("");
  const [busy, setBusy] = useState(false);
  const toast = useToast();

  const create = async () => {
    if (!name.trim()) return;
    setBusy(true);
    try {
      const p = await api.createProject(name.trim(), desc.trim());
      setName("");
      setDesc("");
      onChanged();
      onSelect(p);
    } catch (e) {
      toast((e as Error).message, true);
    } finally {
      setBusy(false);
    }
  };

  const remove = async (p: Project) => {
    if (!confirm(`Delete "${p.name}" and all its sources, samples, and run history? This cannot be undone.`)) return;
    try {
      await api.deleteProject(p.id);
      onChanged();
      toast(`Project "${p.name}" deleted.`);
    } catch (e) {
      toast((e as Error).message, true);
    }
  };

  return (
    <div>
      <h1 className="page-title">Projects</h1>
      <p className="page-sub">
        Each project is an independent training dataset — a separate corpus, pipeline config, and export.
      </p>

      <div className="grid cols-2" style={{ marginBottom: 24, alignItems: "start" }}>
        {/* Create form */}
        <div className="card">
          <h3>New project</h3>
          <label className="field">
            <span>Name <span className="muted">(e.g. Race Car Engineer, Compliance Expert)</span></span>
            <input
              type="text"
              placeholder="Dataset name…"
              value={name}
              onChange={(e) => setName(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && create()}
            />
          </label>
          <label className="field">
            <span>Description <span className="muted">(optional)</span></span>
            <textarea
              rows={2}
              placeholder="What will this model be trained to do?"
              value={desc}
              onChange={(e) => setDesc(e.target.value)}
            />
          </label>
          <button className="btn" onClick={create} disabled={busy || !name.trim()}>
            {busy ? "Creating…" : "Create project"}
          </button>
        </div>

        {/* Quick tips */}
        <div className="card" style={{ background: "transparent", border: "1px dashed var(--border)" }}>
          <h3>How it works</h3>
          <ol style={{ color: "var(--muted)", lineHeight: 1.9, paddingLeft: 18, margin: 0, fontSize: 13 }}>
            <li>Create a project for each domain or persona you want to train</li>
            <li>Add sources — PDFs, Word docs, text files, Markdown, or URLs</li>
            <li>Configure the LLM pipeline (chunking, sample types, provider)</li>
            <li>Run → the app generates Q&A and instruction-following samples</li>
            <li>Review, approve, edit samples; export Alpaca JSONL to your trainer</li>
          </ol>
        </div>
      </div>

      {/* Project list */}
      {projects.length === 0 ? (
        <div className="empty">No projects yet — create one above to get started.</div>
      ) : (
        <div className="grid cols-2">
          {projects.map((p) => (
            <div
              key={p.id}
              className="card"
              style={{ cursor: "pointer", transition: "border-color .15s" }}
              onMouseEnter={(e) => (e.currentTarget.style.borderColor = "var(--accent)")}
              onMouseLeave={(e) => (e.currentTarget.style.borderColor = "var(--border)")}
            >
              <div className="row" style={{ marginBottom: 8 }}>
                <div style={{ flex: 1 }}>
                  <div style={{ fontWeight: 700, fontSize: 16 }}>{p.name}</div>
                  {p.description && (
                    <div className="muted" style={{ fontSize: 12, marginTop: 2 }}>{p.description}</div>
                  )}
                </div>
                <button
                  className="btn small danger"
                  onClick={(e) => { e.stopPropagation(); remove(p); }}
                >
                  Delete
                </button>
              </div>

              <div className="pill-row" style={{ marginBottom: 14 }}>
                <span className="badge">{p.source_count} source{p.source_count !== 1 ? "s" : ""}</span>
                <span className="badge">{p.sample_count} sample{p.sample_count !== 1 ? "s" : ""}</span>
                <span className="badge muted" style={{ fontSize: 10 }}>
                  {new Date(p.created_at).toLocaleDateString()}
                </span>
              </div>

              <button className="btn" onClick={() => onSelect(p)} style={{ width: "100%" }}>
                Open project →
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
