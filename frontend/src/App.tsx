import { useCallback, useEffect, useState } from "react";
import { api } from "./api";
import type { Meta, Project, Stats } from "./types";
import { LLMStatusChip, ToastProvider, useLLMStatus } from "./ui";
import ProjectsPage from "./pages/Projects";
import Dashboard from "./pages/Dashboard";
import Sources from "./pages/Sources";
import Configure from "./pages/Configure";
import RunMonitor from "./pages/RunMonitor";
import Review from "./pages/Review";
import ExportPanel from "./pages/ExportPanel";
import Training from "./pages/Training";

type WorkspaceView = "dashboard" | "sources" | "configure" | "run" | "review" | "export" | "training";

const NAV: { id: WorkspaceView; label: string; icon: string }[] = [
  { id: "dashboard", label: "Dashboard", icon: "▦" },
  { id: "sources", label: "Sources", icon: "▤" },
  { id: "configure", label: "Pipeline", icon: "⚙" },
  { id: "run", label: "Run", icon: "▶" },
  { id: "review", label: "Review", icon: "✓" },
  { id: "export", label: "Export", icon: "⤓" },
  { id: "training", label: "Train (GPU)", icon: "⚡" },
];

function Shell() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [activeProject, setActiveProject] = useState<Project | null>(null);
  const [meta, setMeta] = useState<Meta | null>(null);
  const [stats, setStats] = useState<Stats | null>(null);
  const [view, setView] = useState<WorkspaceView>("dashboard");
  const [loading, setLoading] = useState(true);
  const llmStatus = useLLMStatus(activeProject?.id ?? null);

  const loadProjects = useCallback(async () => {
    const ps = await api.listProjects();
    setProjects(ps);
    return ps;
  }, []);

  const refreshStats = useCallback(async (pid: string) => {
    try { setStats(await api.stats(pid)); } catch { /* ignore */ }
  }, []);

  useEffect(() => {
    Promise.all([api.meta(), loadProjects()]).then(([m]) => {
      setMeta(m);
      setLoading(false);
    });
  }, [loadProjects]);

  // Whenever the active project changes, reload its stats.
  useEffect(() => {
    if (activeProject) refreshStats(activeProject.id);
  }, [activeProject, refreshStats]);

  const selectProject = (p: Project) => {
    setActiveProject(p);
    setView("dashboard");
    setStats(null);
  };

  const handleProjectsChanged = async () => {
    const ps = await loadProjects();
    // If the active project was deleted, go back to project list.
    if (activeProject && !ps.find((p) => p.id === activeProject.id)) {
      setActiveProject(null);
    } else if (activeProject) {
      const updated = ps.find((p) => p.id === activeProject.id);
      if (updated) setActiveProject(updated);
    }
  };

  const onChanged = useCallback(() => {
    if (activeProject) refreshStats(activeProject.id);
  }, [activeProject, refreshStats]);

  if (loading || !meta) {
    return <div className="empty" style={{ marginTop: 80 }}>Loading…</div>;
  }

  // ── Project picker ──────────────────────────────────────────────────────
  if (!activeProject) {
    return (
      <div className="app" style={{ display: "block" }}>
        <div style={{ borderBottom: "1px solid var(--border)", padding: "14px 36px", display: "flex", alignItems: "center", gap: 14, background: "var(--panel)" }}>
          <span style={{ fontWeight: 700, fontSize: 16 }}>LoRA Data Builder</span>
        </div>
        <div className="main" style={{ maxWidth: 900 }}>
          <ProjectsPage
            projects={projects}
            onSelect={selectProject}
            onChanged={handleProjectsChanged}
          />
        </div>
      </div>
    );
  }

  // ── Per-project workspace ────────────────────────────────────────────────
  return (
    <div className="app">
      <nav className="sidebar">
        {/* Project header + switch button */}
        <div style={{ marginBottom: 14 }}>
          <div className="brand">
            LoRA Data Builder
            <small style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
              {activeProject.name}
            </small>
          </div>
          <button
            className="btn small ghost"
            style={{ width: "100%", marginTop: 6, justifyContent: "center" }}
            onClick={() => { setActiveProject(null); setStats(null); }}
          >
            ⇠ All projects
          </button>
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
              <span className="badge amber" style={{ marginLeft: "auto" }}>
                {stats.by_status.pending_review}
              </span>
            ) : null}
          </div>
        ))}

        <div className="spacer" />
        <div style={{ padding: "0 4px", marginBottom: 8 }}>
          <LLMStatusChip status={llmStatus} />
        </div>
        <div className="muted" style={{ fontSize: 11, padding: "0 8px", marginBottom: 4 }}>
          {stats ? `${stats.samples} samples · ${stats.approved_pct}% approved` : ""}
        </div>
      </nav>

      <main className="main">
        {view === "dashboard" && (
          <Dashboard project={activeProject} stats={stats} go={setView} />
        )}
        {view === "sources" && (
          <Sources project={activeProject} onChanged={onChanged} />
        )}
        {view === "configure" && (
          <Configure
            project={activeProject}
            meta={meta}
            onSaved={(p) => { setActiveProject(p); }}
          />
        )}
        {view === "run" && (
          <RunMonitor
            project={activeProject}
            onChanged={onChanged}
            goReview={() => setView("review")}
          />
        )}
        {view === "review" && (
          <Review project={activeProject} stats={stats} onChanged={onChanged} />
        )}
        {view === "export" && (
          <ExportPanel project={activeProject} meta={meta} stats={stats} />
        )}
        {view === "training" && (
          <Training project={activeProject} stats={stats} />
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
