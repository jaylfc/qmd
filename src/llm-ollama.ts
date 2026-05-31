// =============================================================================
// OllamaLLM — an LLM backend backed by an Ollama-compatible HTTP API.
// =============================================================================
//
// Implements the `LLM` interface against an Ollama-compatible server's native
// API (/api/embed, /api/rerank, /api/generate, /api/tags). This lets qmd use
// an existing Ollama install (GPU) or an rkllama NPU server as its model
// provider instead of the bundled llama.cpp. Inject with
// `setDefaultLLM(new OllamaLLM({ url }))` or via `qmd serve --backend ollama`.
//
// No local tokenizer is exposed by the Ollama API, so the optional
// tokenize/detokenize/countTokens members are omitted; consumers fall back to
// character-based handling.

import type {
  LLM,
  EmbedOptions,
  EmbeddingResult,
  GenerateOptions,
  GenerateResult,
  ModelInfo,
  Queryable,
  QueryType,
  RerankDocument,
  RerankOptions,
  RerankResult,
} from "./llm.js";

export interface OllamaLLMOptions {
  /** Base URL of the Ollama-compatible server (default http://localhost:11434). */
  url?: string;
  embedModel?: string;
  rerankModel?: string;
  expandModel?: string;
  timeoutMs?: number;
}

export class OllamaLLM implements LLM {
  private readonly base: string;
  private readonly _embedModel: string;
  private readonly _rerankModel: string;
  private readonly _expandModel: string;
  private readonly timeoutMs: number;

  constructor(options: OllamaLLMOptions | string = {}) {
    const opts = typeof options === "string" ? { url: options } : options;
    this.base = (opts.url ?? "http://localhost:11434").replace(/\/+$/, "");
    this._embedModel = opts.embedModel ?? "qwen3-embedding-0.6b";
    this._rerankModel = opts.rerankModel ?? "qwen3-reranker-0.6b";
    this._expandModel = opts.expandModel ?? "qmd-query-expansion";
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
        throw new Error(`ollama ${path} -> HTTP ${res.status}: ${await res.text().catch(() => "")}`);
      }
      return (await res.json()) as T;
    } finally {
      clearTimeout(timer);
    }
  }

  // ── LLM interface ──────────────────────────────────────────────────────

  async embed(text: string, options?: EmbedOptions): Promise<EmbeddingResult | null> {
    if (!text) return null;
    // Qwen3-Embedding uses an instruction prefix for queries.
    const input = options?.isQuery
      ? `Instruct: Retrieve relevant documents for the given query\nQuery: ${text}`
      : text;
    const r = await this.post<{ embeddings?: number[][] }>("/api/embed", { model: this._embedModel, input });
    const vec = r.embeddings?.[0];
    if (!vec) return null;
    return { embedding: vec, model: this._embedModel };
  }

  async embedBatch(texts: string[], options?: EmbedOptions): Promise<(EmbeddingResult | null)[]> {
    if (texts.length === 0) return [];
    const input = options?.isQuery
      ? texts.map((t) => `Instruct: Retrieve relevant documents for the given query\nQuery: ${t}`)
      : texts;
    try {
      // Ollama /api/embed accepts an array and returns one embedding per input.
      const r = await this.post<{ embeddings?: number[][] }>("/api/embed", { model: this._embedModel, input });
      if (!r.embeddings) return texts.map(() => null);
      return r.embeddings.map((emb) => (emb ? { embedding: emb, model: this._embedModel } : null));
    } catch {
      // Fall back to individual embeds if the batch call fails.
      const out: (EmbeddingResult | null)[] = [];
      for (const t of texts) out.push(await this.embed(t, options));
      return out;
    }
  }

  async generate(prompt: string, options?: GenerateOptions): Promise<GenerateResult | null> {
    const r = await this.post<{ response?: string; done?: boolean }>("/api/generate", {
      model: options?.model ?? this._expandModel,
      prompt,
      stream: false,
      options: {
        temperature: options?.temperature ?? 0.7,
        num_predict: options?.maxTokens ?? 600,
      },
    });
    if (typeof r.response !== "string") return null;
    return { text: r.response, model: options?.model ?? this._expandModel, done: r.done ?? true };
  }

  async rerank(query: string, documents: RerankDocument[], _options?: RerankOptions): Promise<RerankResult> {
    if (documents.length === 0) return { results: [], model: this._rerankModel };
    const r = await this.post<{ results: { index: number; relevance_score: number }[] }>("/api/rerank", {
      model: this._rerankModel,
      query,
      documents: documents.map((d) => d.text),
    });
    const results = r.results.map((x) => ({
      file: documents[x.index]?.file ?? `doc-${x.index}`,
      score: x.relevance_score,
      index: x.index,
    }));
    results.sort((a, b) => b.score - a.score);
    return { results, model: this._rerankModel };
  }

  async expandQuery(
    query: string,
    options?: { context?: string; includeLexical?: boolean; intent?: string },
  ): Promise<Queryable[]> {
    if (!query) return [];
    const prompt = options?.intent
      ? `/no_think Expand this search query: ${query}\nQuery intent: ${options.intent}`
      : `/no_think Expand this search query: ${query}`;
    let response: string;
    try {
      const r = await this.post<{ response?: string }>("/api/generate", {
        model: this._expandModel,
        prompt,
        stream: false,
        options: { temperature: 0.7, top_k: 20, top_p: 0.8, num_predict: 600 },
      });
      response = (r.response ?? "").trim();
    } catch {
      return [];
    }

    const includeLexical = options?.includeLexical ?? true;
    const queryTerms = query.toLowerCase().replace(/[^a-z0-9\s]/g, " ").split(/\s+/).filter(Boolean);
    const hasQueryTerm = (text: string): boolean => {
      if (queryTerms.length === 0) return true;
      const lower = text.toLowerCase();
      return queryTerms.some((t) => lower.includes(t));
    };

    const queryables: Queryable[] = [];
    for (const line of response.split("\n")) {
      const trimmed = line.trim();
      const colon = trimmed.indexOf(":");
      if (colon === -1) continue;
      const type = trimmed.slice(0, colon).trim().toLowerCase();
      if (type !== "lex" && type !== "vec" && type !== "hyde") continue;
      const text = trimmed.slice(colon + 1).trim();
      if (!text || !hasQueryTerm(text)) continue;
      queryables.push({ type: type as QueryType, text });
    }

    const filtered = includeLexical ? queryables : queryables.filter((q) => q.type !== "lex");
    if (filtered.length > 0) return filtered;

    // Fallback: model produced free text instead of structured lines.
    const fallback: Queryable[] = [
      { type: "hyde", text: `Information about ${query}` },
      { type: "lex", text: query },
      { type: "vec", text: query },
    ];
    return includeLexical ? fallback : fallback.filter((q) => q.type !== "lex");
  }

  async modelExists(model: string): Promise<ModelInfo> {
    try {
      const res = await fetch(this.base + "/api/tags");
      const data = (await res.json()) as { models?: { name: string }[] };
      const names = (data.models ?? []).map((m) => m.name);
      return { name: model, exists: names.includes(model) };
    } catch {
      return { name: model, exists: false };
    }
  }

  async dispose(): Promise<void> {
    // External process — nothing to dispose.
  }

  get embedModelName(): string { return this._embedModel; }
  get generateModelName(): string { return this._expandModel; }
  get rerankModelName(): string { return this._rerankModel; }

  // No tokenize/detokenize/countTokens — Ollama exposes no tokenizer; consumers
  // fall back to character-based handling.
}
