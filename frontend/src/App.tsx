import { useCallback, useEffect, useState } from "react";
import { api } from "./api";
import type { Meta, Project, Stats } from "./types";
import { ToastProvider } from "./ui";
import Dashboard from "./pages/Dashboard";
import Sources from "./pages/Sources";
import Configure from "./pages/Configure";
import RunMonitor from "./pages/RunMonitor";
import Review from "./pages/Review";
import ExportPanel from "./pages/ExportPanel";

type View = "dashboard" | "sources" | "configure" | "run" | "review" | "export";

const NAV: { id: View; label: string; icon: string }[] = [
  { id: "dashboard", label: "Dashboard", icon: "▦" },
  { id: "sources", label: "Sources", icon: "▤" },
  { id: "configure", label: "Pipeline", icon: "⚙" },
  { id: "run", label: "Run", icon: "▶" },
  { id: "review", label: "Review", icon: "✓" },
  { id: "export", label: "Export", icon: "⤓" },
];

function Shell() {
  const [view, setView] = useState<View>("dashboard");
  const [project, setProject] = useState<Project | null>(null);
  const [meta, setMeta] = useState<Meta | null>(null);
  const [stats, setStats] = useState<Stats | null>(null);

  const loadProject = useCallback(async () => {
    const projects = await api.listProjects();
    setProject(projects[0]);
    return projects[0];
  }, []);

  const refreshStats = useCallback(async (pid: string) => {
    try {
      setStats(await api.stats(pid));
    } catch {
      /* ignore */
    }
  }, []);

  useEffect(() => {
    api.meta().then(setMeta);
    loadProject().then((p) => p && refreshStats(p.id));
  }, [loadProject, refreshStats]);

  if (!project || !meta) {
    return <div className="empty">Loading…</div>;
  }

  const onChanged = () => refreshStats(project.id);

  return (
    <div className="app">
      <nav className="sidebar">
        <div className="brand">
          LoRA Data Builder
          <small>{project.name}</small>
        </div>
        {NAV.map((n) => (
          <div
            key={n.id}
            className={`nav-item ${view === n.id ? "active" : ""}`}
            onClick={() => setView(n.id)}
          >
            <span style={{ width: 16, textAlign: "center" }}>{n.icon}</span>
            {n.label}
            {n.id === "review" && stats && stats.by_status.pending_review ? (
              <span className="badge amber">{stats.by_status.pending_review}</span>
            ) : null}
          </div>
        ))}
        <div className="spacer" />
        <div className="muted" style={{ fontSize: 11, padding: "0 8px" }}>
          {stats ? `${stats.samples} samples · ${stats.approved_pct}% approved` : ""}
        </div>
      </nav>

      <main className="main">
        {view === "dashboard" && (
          <Dashboard project={project} stats={stats} go={setView} />
        )}
        {view === "sources" && (
          <Sources project={project} onChanged={onChanged} />
        )}
        {view === "configure" && (
          <Configure project={project} meta={meta} onSaved={setProject} />
        )}
        {view === "run" && (
          <RunMonitor project={project} onChanged={onChanged} goReview={() => setView("review")} />
        )}
        {view === "review" && (
          <Review project={project} stats={stats} onChanged={onChanged} />
        )}
        {view === "export" && (
          <ExportPanel project={project} meta={meta} stats={stats} />
        )}
      </main>
    </div>
  );
}

export default function App() {
  return (
    <ToastProvider>
      <Shell />
    </ToastProvider>
  );
}
