import type {
  Chunk, DryRun, ExportResult, Meta, PipelineConfig, Project, Run, Sample, Source, Stats,
} from "./types";

const BASE = "/api";

async function req<T>(path: string, opts: RequestInit = {}): Promise<T> {
  const res = await fetch(BASE + path, {
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || JSON.stringify(body);
    } catch {
      /* ignore */
    }
    throw new Error(detail);
  }
  if (res.status === 204) return undefined as T;
  return res.json() as Promise<T>;
}

export const api = {
  meta: () => req<Meta>("/meta"),

  // Projects
  listProjects: () => req<Project[]>("/projects"),
  getProject: (id: string) => req<Project>(`/projects/${id}`),
  createProject: (name: string, description = "") =>
    req<Project>("/projects", {
      method: "POST",
      body: JSON.stringify({ name, description }),
    }),
  updateProject: (id: string, body: { name?: string; description?: string; config?: Partial<PipelineConfig> }) =>
    req<Project>(`/projects/${id}`, { method: "PATCH", body: JSON.stringify(body) }),
  deleteProject: (id: string) =>
    req<{ ok: boolean }>(`/projects/${id}`, { method: "DELETE" }),

  // Sources
  listSources: (pid: string) => req<Source[]>(`/projects/${pid}/sources`),
  addUrl: (pid: string, url: string, title?: string) =>
    req<Source>(`/projects/${pid}/sources/url`, {
      method: "POST",
      body: JSON.stringify({ url, title }),
    }),
  uploadFiles: async (pid: string, files: FileList) => {
    const form = new FormData();
    Array.from(files).forEach((f) => form.append("files", f));
    const res = await fetch(`${BASE}/projects/${pid}/sources/upload`, {
      method: "POST",
      body: form,
    });
    if (!res.ok) throw new Error((await res.json()).detail || "Upload failed");
    return res.json() as Promise<Source[]>;
  },
  deleteSource: (pid: string, sid: string) =>
    req<{ ok: boolean }>(`/projects/${pid}/sources/${sid}`, { method: "DELETE" }),
  sourceChunks: (pid: string, sid: string) =>
    req<Chunk[]>(`/projects/${pid}/sources/${sid}/chunks`),

  // Runs
  dryRun: (pid: string) => req<DryRun>(`/projects/${pid}/runs/dry-run`),
  startRun: (pid: string) => req<Run>(`/projects/${pid}/runs/start`, { method: "POST" }),
  listRuns: (pid: string) => req<Run[]>(`/projects/${pid}/runs`),
  getRun: (pid: string, rid: string) => req<Run>(`/projects/${pid}/runs/${rid}`),
  cancelRun: (pid: string, rid: string) =>
    req<{ cancelling: boolean }>(`/projects/${pid}/runs/${rid}/cancel`, { method: "POST" }),

  // Samples
  listSamples: (pid: string, params: Record<string, string> = {}) => {
    const qs = new URLSearchParams(params).toString();
    return req<Sample[]>(`/projects/${pid}/samples${qs ? `?${qs}` : ""}`);
  },
  updateSample: (pid: string, sid: string, body: Partial<Sample>) =>
    req<Sample>(`/projects/${pid}/samples/${sid}`, {
      method: "PATCH",
      body: JSON.stringify(body),
    }),
  bulk: (pid: string, action: string, ids?: string[]) =>
    req<{ changed: number }>(`/projects/${pid}/samples/bulk`, {
      method: "POST",
      body: JSON.stringify({ action, ids }),
    }),
  stats: (pid: string) => req<Stats>(`/projects/${pid}/samples/stats`),

  // Export
  exportDataset: (pid: string, body: { format: string; train_split: number; include_statuses: string[] }) =>
    req<ExportResult>(`/projects/${pid}/export`, { method: "POST", body: JSON.stringify(body) }),
  downloadUrl: (pid: string, path: string) =>
    `${BASE}/projects/${pid}/export/download?path=${encodeURIComponent(path)}`,
};
