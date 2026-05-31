// =============================================================================
// qmd serve — an HTTP model server.
// =============================================================================
//
// Exposes the active LLM backend (whatever `getDefaultLLM()` resolves to) over
// a small HTTP API so thin clients can offload embedding / reranking / query
// expansion to one machine — e.g. an SBC with an NPU serving lighter clients.
// Built on the pluggable-backend seam; uses only `node:http` (no new deps).
//
// API (stable; matches what existing qmd clients expect):
//   GET  /health  -> { ok, version, backend, models: { embed, rerank, generate } }
//   GET  /status  -> status payload (when a statusProvider is supplied)
//   POST /embed   -> { text, isQuery? }                 -> { embedding, model }
//   POST /rerank  -> { query, documents: [{file,text}] } -> { results, model }
//   POST /expand  -> { query }                           -> { queries }

import { createServer, type Server, type IncomingMessage, type ServerResponse } from "node:http";
import { getDefaultLLM } from "./llm.js";

const SERVE_API_VERSION = "2";

export interface ServeOptions {
  port: number;
  /** Bind address, e.g. "0.0.0.0" or "127.0.0.1". */
  bind?: string;
  /** Optional backend label reported by /health (e.g. "local", "rkllama"). */
  backend?: string;
  /** Optional provider for the /status payload. */
  statusProvider?: () => unknown | Promise<unknown>;
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const payload = JSON.stringify(body);
  res.writeHead(status, { "Content-Type": "application/json" });
  res.end(payload);
}

async function readJsonBody(req: IncomingMessage): Promise<Record<string, unknown>> {
  const chunks: Buffer[] = [];
  for await (const chunk of req) chunks.push(chunk as Buffer);
  if (chunks.length === 0) return {};
  const raw = Buffer.concat(chunks).toString("utf8").trim();
  if (!raw) return {};
  const parsed = JSON.parse(raw);
  if (typeof parsed !== "object" || parsed === null) throw new Error("body must be a JSON object");
  return parsed as Record<string, unknown>;
}

async function handleEmbed(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJsonBody(req);
  const text = body.text;
  if (typeof text !== "string" || text.length === 0) {
    return sendJson(res, 400, { error: "text must be a non-empty string" });
  }
  const result = await getDefaultLLM().embed(text, { isQuery: body.isQuery === true });
  if (!result) return sendJson(res, 500, { error: "embedding failed" });
  sendJson(res, 200, { embedding: result.embedding, model: result.model });
}

async function handleRerank(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJsonBody(req);
  const query = body.query;
  const documents = body.documents;
  if (typeof query !== "string" || query.length === 0 || !Array.isArray(documents) || documents.length === 0) {
    return sendJson(res, 400, { error: "query must be a non-empty string and documents must be a non-empty array" });
  }
  const docs = documents.map((d, i) => {
    if (typeof d === "string") return { file: String(i), text: d };
    const obj = d as { file?: unknown; text?: unknown };
    if (typeof obj.text !== "string") throw new Error("each document must be a string or { file, text }");
    return { file: typeof obj.file === "string" ? obj.file : String(i), text: obj.text };
  });
  const result = await getDefaultLLM().rerank(query, docs);
  sendJson(res, 200, { results: result.results, model: result.model });
}

async function handleExpand(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJsonBody(req);
  const query = body.query;
  if (typeof query !== "string" || query.length === 0) {
    return sendJson(res, 400, { error: "query must be a non-empty string" });
  }
  const queries = await getDefaultLLM().expandQuery(query);
  sendJson(res, 200, { queries });
}

function handleHealth(res: ServerResponse, backend: string): void {
  const llm = getDefaultLLM();
  sendJson(res, 200, {
    ok: true,
    version: SERVE_API_VERSION,
    backend,
    models: {
      embed: llm.embedModelName,
      rerank: llm.rerankModelName,
      generate: llm.generateModelName,
    },
  });
}

/**
 * Start the qmd HTTP model server. Returns the listening server so callers can
 * close it (used by tests).
 */
export function startServer(options: ServeOptions): Promise<Server> {
  const bind = options.bind ?? "127.0.0.1";
  const backend = options.backend ?? "local";

  const server = createServer((req, res) => {
    const method = req.method ?? "GET";
    const path = (req.url ?? "/").split("?")[0];

    void (async () => {
      try {
        if (method === "GET" && path === "/health") return handleHealth(res, backend);
        if (method === "GET" && path === "/status") {
          if (!options.statusProvider) return sendJson(res, 200, { ok: true });
          return sendJson(res, 200, await options.statusProvider());
        }
        if (method === "POST" && path === "/embed") return await handleEmbed(req, res);
        if (method === "POST" && path === "/rerank") return await handleRerank(req, res);
        if (method === "POST" && path === "/expand") return await handleExpand(req, res);
        if (method !== "GET" && method !== "POST") return sendJson(res, 405, { error: "Method not allowed" });
        sendJson(res, 404, { error: "Not found" });
      } catch (err) {
        sendJson(res, 400, { error: err instanceof Error ? err.message : String(err) });
      }
    })();
  });

  return new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(options.port, bind, () => {
      server.removeListener("error", reject);
      resolve(server);
    });
  });
}
