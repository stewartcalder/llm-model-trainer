import { Fragment, useEffect, useRef, useState } from "react";
import { api } from "../api";
import type { Chunk, Project, Source } from "../types";
import { Badge, useToast } from "../ui";

export default function Sources({ project, onChanged }: { project: Project; onChanged: () => void }) {
  const [sources, setSources] = useState<Source[]>([]);
  const [url, setUrl] = useState("");
  const [drag, setDrag] = useState(false);
  const [busy, setBusy] = useState(false);
  const [chunksFor, setChunksFor] = useState<string | null>(null);
  const [chunks, setChunks] = useState<Chunk[]>([]);
  const fileRef = useRef<HTMLInputElement>(null);
  const toast = useToast();

  const load = () => api.listSources(project.id).then(setSources);
  useEffect(() => { load(); }, [project.id]);

  const addUrl = async () => {
    if (!url.trim()) return;
    setBusy(true);
    try {
      await api.addUrl(project.id, url.trim());
      setUrl("");
      await load();
      onChanged();
      toast("URL added. It will be fetched on the next run.");
    } catch (e) {
      toast((e as Error).message, true);
    } finally {
      setBusy(false);
    }
  };

  const upload = async (files: FileList | null) => {
    if (!files || files.length === 0) return;
    setBusy(true);
    try {
      const added = await api.uploadPdfs(project.id, files);
      await load();
      onChanged();
      toast(`${added.length} PDF(s) added${added.length === 0 ? " (duplicates skipped)" : ""}.`);
    } catch (e) {
      toast((e as Error).message, true);
    } finally {
      setBusy(false);
    }
  };

  const remove = async (s: Source) => {
    if (!confirm(`Remove "${s.title || s.path_or_url}" and its chunks/samples?`)) return;
    await api.deleteSource(project.id, s.id);
    await load();
    onChanged();
  };

  const viewChunks = async (s: Source) => {
    if (chunksFor === s.id) { setChunksFor(null); return; }
    setChunks(await api.sourceChunks(project.id, s.id));
    setChunksFor(s.id);
  };

  return (
    <div>
      <h1 className="page-title">Sources</h1>
      <p className="page-sub">Add PDFs and URLs. Text is extracted and chunked when you run the pipeline.</p>

      <div className="grid cols-2" style={{ marginBottom: 18 }}>
        <div
          className={`dropzone ${drag ? "drag" : ""}`}
          onClick={() => fileRef.current?.click()}
          onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
          onDragLeave={() => setDrag(false)}
          onDrop={(e) => { e.preventDefault(); setDrag(false); upload(e.dataTransfer.files); }}
        >
          <div style={{ fontSize: 22, marginBottom: 6 }}>⤓</div>
          <div><strong>Drop PDF files</strong> or click to browse</div>
          <div style={{ fontSize: 12, marginTop: 4 }}>Duplicate files are skipped automatically</div>
          <input
            ref={fileRef}
            type="file"
            accept="application/pdf"
            multiple
            style={{ display: "none" }}
            onChange={(e) => upload(e.target.files)}
          />
        </div>

        <div className="card">
          <h3>Add a URL</h3>
          <input
            type="url"
            placeholder="https://example.com/article"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && addUrl()}
          />
          <div style={{ marginTop: 12 }}>
            <button className="btn" onClick={addUrl} disabled={busy}>Add URL</button>
          </div>
        </div>
      </div>

      <div className="card">
        <h3>Source list ({sources.length})</h3>
        {sources.length === 0 ? (
          <div className="muted">No sources yet.</div>
        ) : (
          <table>
            <thead>
              <tr>
                <th>Title</th>
                <th>Type</th>
                <th>Status</th>
                <th>Chunks</th>
                <th>Samples</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {sources.map((s) => (
                <Fragment key={s.id}>
                  <tr>
                    <td>
                      <div>{s.title || <span className="muted">(untitled)</span>}</div>
                      <div className="mono muted" style={{ fontSize: 11 }}>{s.path_or_url}</div>
                      {s.error && <div className="badge red" style={{ marginTop: 4 }}>{s.error}</div>}
                    </td>
                    <td><span className="badge blue">{s.type}</span></td>
                    <td><Badge status={s.status} /></td>
                    <td>{s.chunk_count}</td>
                    <td>{s.sample_count}</td>
                    <td>
                      <div className="row">
                        {s.chunk_count > 0 && (
                          <button className="btn small ghost" onClick={() => viewChunks(s)}>
                            {chunksFor === s.id ? "Hide" : "Chunks"}
                          </button>
                        )}
                        <button className="btn small danger" onClick={() => remove(s)}>Remove</button>
                      </div>
                    </td>
                  </tr>
                  {chunksFor === s.id && (
                    <tr>
                      <td colSpan={6} style={{ background: "var(--bg)" }}>
                        {chunks.map((c, i) => (
                          <div key={c.id} style={{ marginBottom: 8 }}>
                            <span className="badge">#{i + 1} · {c.page_or_section} · {c.token_count} tok</span>
                            <div className="muted" style={{ marginTop: 4, fontSize: 12 }}>
                              {c.text.slice(0, 260)}{c.text.length > 260 ? "…" : ""}
                            </div>
                          </div>
                        ))}
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
