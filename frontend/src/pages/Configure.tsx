import { useState } from "react";
import { api } from "../api";
import type { Meta, PipelineConfig, Project } from "../types";
import { useToast } from "../ui";

interface Props {
  project: Project;
  meta: Meta;
  onSaved: (p: Project) => void;
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="card" style={{ marginBottom: 14 }}>
      <h3>{title}</h3>
      {children}
    </div>
  );
}

export default function Configure({ project, meta, onSaved }: Props) {
  const [cfg, setCfg] = useState<PipelineConfig>(structuredClone(project.config));
  const [name, setName] = useState(project.name);
  const [desc, setDesc] = useState(project.description);
  const [saving, setSaving] = useState(false);
  const [editingTemplates, setEditingTemplates] = useState(false);
  const toast = useToast();

  const set = (path: string, value: unknown) => {
    setCfg((prev) => {
      const next = structuredClone(prev);
      const keys = path.split(".");
      let obj: any = next;
      for (let i = 0; i < keys.length - 1; i++) obj = obj[keys[i]];
      obj[keys[keys.length - 1]] = value;
      return next;
    });
  };

  const toggleType = (id: string) => {
    const has = cfg.sample_types.includes(id);
    set("sample_types", has ? cfg.sample_types.filter((t) => t !== id) : [...cfg.sample_types, id]);
  };

  const save = async () => {
    setSaving(true);
    try {
      const updated = await api.updateProject(project.id, { name, description: desc, config: cfg });
      onSaved(updated);
      toast("Pipeline configuration saved.");
    } catch (e) {
      toast((e as Error).message, true);
    } finally {
      setSaving(false);
    }
  };

  const provider = cfg.llm.provider;

  return (
    <div>
      <h1 className="page-title">Pipeline Configuration</h1>
      <p className="page-sub">Settings are saved with the project and snapshotted into each run for reproducibility.</p>

      <Section title="Project">
        <label className="field">
          <span>Project name</span>
          <input type="text" value={name} onChange={(e) => setName(e.target.value)} />
        </label>
        <label className="field">
          <span>Description</span>
          <textarea rows={2} value={desc} onChange={(e) => setDesc(e.target.value)}
            placeholder="What will this model be trained to do?" />
        </label>
      </Section>

      <Section title="Chunking (sentence-window)">
        <div className="grid cols-2">
          <label className="field">
            <span>Window size (tokens)</span>
            <input type="number" value={cfg.chunking.window_tokens}
              onChange={(e) => set("chunking.window_tokens", +e.target.value)} />
          </label>
          <label className="field">
            <span>Overlap (tokens)</span>
            <input type="number" value={cfg.chunking.overlap_tokens}
              onChange={(e) => set("chunking.overlap_tokens", +e.target.value)} />
          </label>
        </div>
      </Section>

      <Section title="Sample types">
        <div className="pill-row">
          {meta.sample_types.map((t) => (
            <label key={t.id} className="checkbox">
              <input type="checkbox" checked={cfg.sample_types.includes(t.id)}
                onChange={() => toggleType(t.id)} />
              {t.label}
            </label>
          ))}
        </div>
        <p className="muted" style={{ marginTop: 10, fontSize: 12 }}>
          One LLM call is made per selected type per chunk.
        </p>
      </Section>

      <Section title="LLM provider">
        <div className="grid cols-2">
          <label className="field">
            <span>Provider</span>
            <select value={provider} onChange={(e) => set("llm.provider", e.target.value)}>
              {meta.providers.map((p) => <option key={p} value={p}>{p}</option>)}
            </select>
          </label>
          <label className="field">
            <span>Model</span>
            <input type="text" value={cfg.llm.model} onChange={(e) => set("llm.model", e.target.value)} />
          </label>
        </div>
        {provider === "mock" && (
          <p className="muted" style={{ fontSize: 12 }}>
            The <strong>mock</strong> provider runs fully offline with no API key — ideal for a first end-to-end test.
          </p>
        )}
        {provider === "anthropic" && (
          <label className="field">
            <span>API key (leave blank to use ANTHROPIC_API_KEY env var)</span>
            <input type="text" placeholder="sk-ant-…" value={cfg.llm.api_key || ""}
              onChange={(e) => set("llm.api_key", e.target.value)} />
          </label>
        )}
        {provider === "ollama" && (
          <label className="field">
            <span>Base URL (OpenAI-compatible endpoint)</span>
            <input type="text" value={cfg.llm.base_url || ""}
              onChange={(e) => set("llm.base_url", e.target.value)} />
          </label>
        )}
        <div className="grid cols-3">
          <label className="field">
            <span>Temperature</span>
            <input type="number" step="0.1" value={cfg.llm.temperature}
              onChange={(e) => set("llm.temperature", +e.target.value)} />
          </label>
          <label className="field">
            <span>Max tokens</span>
            <input type="number" value={cfg.llm.max_tokens}
              onChange={(e) => set("llm.max_tokens", +e.target.value)} />
          </label>
          <label className="field" style={{ display: "flex", alignItems: "flex-end" }}>
            <label className="checkbox">
              <input type="checkbox" checked={cfg.llm.use_critic}
                onChange={(e) => set("llm.use_critic", e.target.checked)} />
              Use critic (self-review)
            </label>
          </label>
        </div>
      </Section>

      <Section title="Concurrency & budget">
        <div className="grid cols-2">
          <label className="field">
            <span>Parallel LLM calls</span>
            <input type="number" value={cfg.concurrency}
              onChange={(e) => set("concurrency", +e.target.value)} />
          </label>
          <label className="field">
            <span>Budget limit (USD, hard stop)</span>
            <input type="number" step="0.5" value={cfg.budget_usd}
              onChange={(e) => set("budget_usd", +e.target.value)} />
          </label>
        </div>
      </Section>

      <Section title="Prompt templates">
        <div className="accordion-h" onClick={() => setEditingTemplates(!editingTemplates)}>
          {editingTemplates ? "▾" : "▸"} Edit generation & critic templates
        </div>
        {editingTemplates && (
          <div style={{ marginTop: 12 }}>
            {Object.entries(cfg.prompt_templates || {}).map(([key, val]) => (
              <label className="field" key={key}>
                <span>{key} <span className="muted">— variables like {"{chunk_text}"} are substituted</span></span>
                <textarea value={val} rows={5}
                  onChange={(e) => set(`prompt_templates.${key}`, e.target.value)} />
              </label>
            ))}
          </div>
        )}
      </Section>

      <div className="row">
        <button className="btn" onClick={save} disabled={saving}>
          {saving ? "Saving…" : "Save configuration"}
        </button>
      </div>
    </div>
  );
}
