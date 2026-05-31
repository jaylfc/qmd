// =============================================================================
// RemoteLLM — an LLM backend that talks to a remote qmd model server.
// =============================================================================
//
// Implements the `LLM` interface (see llm.ts) over HTTP against a `qmd serve`
// endpoint, so a qmd client can run with no local model/runtime — embeddings,
// reranking and query expansion are served by another machine (e.g. an SBC
// with an NPU). Inject it with `setDefaultLLM(new RemoteLLM(url))` or pass it
// to `createStore({ llm })`.
//
// This backend has NO local tokenizer, so it deliberately omits the optional
// `tokenize`/`detokenize`/`countTokens` members of the `LLM` interface;
// consumers (chunk truncation) fall back to character-based handling.

import type {
  LLM,
  EmbedOptions,
  EmbeddingResult,
  GenerateOptions,
  GenerateResult,
  ModelInfo,
  Queryable,
  RerankDocument,
  RerankOptions,
  RerankResult,
} from "./llm.js";

export interface RemoteLLMOptions {
  /** Base URL of the qmd serve endpoint, e.g. "http://192.168.6.123:7832". */
  url: string;
  /** Per-request timeout in ms (default 60s). */
  timeoutMs?: number;
}

interface HealthResponse {
  ok: boolean;
  version: string;
  backend: string;
  models?: { embed?: string; rerank?: string; generate?: string };
}

export class RemoteLLM implements LLM {
  private readonly base: string;
  private readonly timeoutMs: number;
  private models: HealthResponse["models"] = {};
  private healthPromise: Promise<void> | null = null;

  constructor(options: RemoteLLMOptions | string) {
    const opts = typeof options === "string" ? { url: options } : options;
    this.base = opts.url.replace(/\/+$/, "");
    this.timeoutMs = opts.timeoutMs ?? 60_000;
  }

  private async post<T>(path: string, body: unknown): Promise<T> {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeoutMs);
    try {
      const res = await fetch(this.base + path, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal: controller.signal,
      });
      if (!res.ok) {
        throw new Error(`qmd serve ${path} -> HTTP ${res.status}: ${await res.text()}`);
      }
      return (await res.json()) as T;
    } finally {
      clearTimeout(timer);
    }
  }

  /** Lazily fetch /health once to learn the server's model names. */
  private async ensureHealth(): Promise<void> {
    if (!this.healthPromise) {
      this.healthPromise = (async () => {
        const res = await fetch(this.base + "/health");
        if (res.ok) {
          const h = (await res.json()) as HealthResponse;
          this.models = h.models ?? {};
        }
      })().catch(() => { /* leave models empty; getters fall back to placeholders */ });
    }
    return this.healthPromise;
  }

  // ── LLM interface ──────────────────────────────────────────────────────

  async embed(text: string, options?: EmbedOptions): Promise<EmbeddingResult | null> {
    if (!text) return null;
    const d = await this.post<{ embedding?: number[]; model?: string }>("/embed", {
      text,
      isQuery: options?.isQuery ?? false,
    });
    if (!d.embedding) return null;
    return { embedding: d.embedding, model: d.model ?? this.embedModelName };
  }

  async embedBatch(texts: string[], options?: EmbedOptions): Promise<(EmbeddingResult | null)[]> {
    // The serve protocol has no batch endpoint — issue sequential /embed calls
    // against the single remote backend.
    const out: (EmbeddingResult | null)[] = [];
    for (const t of texts) out.push(await this.embed(t, options));
    return out;
  }

  async generate(_prompt: string, _options?: GenerateOptions): Promise<GenerateResult | null> {
    // Text generation is not part of the qmd serve protocol; return null so
    // callers treat generation as unavailable on this backend.
    return null;
  }

  async modelExists(model: string): Promise<ModelInfo> {
    await this.ensureHealth();
    const known = Object.values(this.models ?? {}).filter(Boolean) as string[];
    return { name: model, exists: known.includes(model) };
  }

  async expandQuery(
    query: string,
    _options?: { context?: string; includeLexical?: boolean; intent?: string },
  ): Promise<Queryable[]> {
    if (!query) return [];
    try {
      const d = await this.post<{ queries?: unknown[]; expansions?: unknown[] }>("/expand", { query });
      const raw = d.queries ?? d.expansions ?? [];
      return raw
        .map((q): Queryable => ({
          type: "vec",
          text: typeof q === "string" ? q : String((q as { text?: unknown }).text ?? ""),
        }))
        .filter((q) => q.text.length > 0);
    } catch {
      // Query expansion is best-effort; on failure fall back to no expansions.
      return [];
    }
  }

  async rerank(
    query: string,
    documents: RerankDocument[],
    _options?: RerankOptions,
  ): Promise<RerankResult> {
    if (documents.length === 0) return { results: [], model: this.rerankModelName };
    const d = await this.post<RerankResult>("/rerank", {
      query,
      documents: documents.map((doc, i) => ({ file: doc.file ?? String(i), text: doc.text })),
    });
    return { results: d.results ?? [], model: d.model ?? this.rerankModelName };
  }

  async dispose(): Promise<void> {
    // No local resources to release.
  }

  get embedModelName(): string {
    return this.models?.embed ?? "remote:embed";
  }

  get generateModelName(): string {
    return this.models?.generate ?? "remote:generate";
  }

  get rerankModelName(): string {
    return this.models?.rerank ?? "remote:rerank";
  }

  // NOTE: tokenize / detokenize / countTokens intentionally omitted — this
  // backend has no local tokenizer. Consumers fall back to char-based handling.
}
