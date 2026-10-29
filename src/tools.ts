import { createHash } from "node:crypto";
import { promises as fs } from "node:fs";
import { resolve, sep } from "node:path";
import { z } from "zod";
import type { CallToolResult } from "@modelcontextprotocol/sdk/types.js";
import type { RequestHandlerExtra } from "@modelcontextprotocol/sdk/shared/protocol.js";
import type { ServerRequest, ServerNotification } from "@modelcontextprotocol/sdk/types.js";
import { chunkExtracted } from "./chunker.js";
import type { Config } from "./config.js";
import { EmptyIndexError, SmartDocsError } from "./errors.js";
import { defaultExtensions, extractBytes, fileKindFor } from "./extract.js";
import { Services } from "./services.js";
import { VectorStore, type StoreIdentity } from "./store.js";
import type { FileKind, SearchHit } from "./types.js";
import { askWithLlm, summarizeWithLlm } from "./llm.js";
import { clipTotalPassages, extractiveSummary, type SummaryPassage } from "./summarize.js";

type Extra = RequestHandlerExtra<ServerRequest, ServerNotification>;

/* --------------------------------- helpers ---------------------------------- */

function jsonResult(value: unknown, isError = false): CallToolResult {
  return {
    content: [{ type: "text", text: JSON.stringify(value, null, 2) }],
    ...(isError ? { isError: true } : {}),
  };
}

function errorResult(err: unknown): CallToolResult {
  if (err instanceof SmartDocsError) {
    return jsonResult({ error: err.name, message: err.message, hint: err.hint }, true);
  }
  const msg = err instanceof Error ? (err.stack ?? err.message) : String(err);
  return jsonResult({ error: "UnexpectedError", message: msg }, true);
}

/** Best-effort progress notifications; never fail the tool because of them. */
async function reportProgress(
  extra: Extra,
  p: { done: number; total: number; currentFile?: string },
): Promise<void> {
  try {
    const token = extra._meta?.progressToken ?? extra.requestId;
    await extra.sendNotification({
      method: "notifications/progress",
      params: {
        progressToken: token,
        progress: p.done,
        ...(p.total > 0 ? { total: p.total } : {}),
        ...(p.currentFile ? { message: `smart-docs: Indexed ${p.currentFile}` } : {}),
      },
    });
  } catch {
    /* advisory only */
  }
}

async function reportLog(extra: Extra, level: "info" | "warning", data: string): Promise<void> {
  try {
    await extra.sendNotification({ method: "notifications/message", params: { level, data, logger: "smart-docs" } });
  } catch {
    /* advisory only */
  }
}

/** Per-run embedding cache so identical chunk texts (across files) embed once. */
class EmbedCache {
  private map = new Map<string, Float32Array>();

  constructor(private readonly provider: { embed(texts: string[]): Promise<Float32Array[]> }) {}

  async embedAll(texts: string[]): Promise<Float32Array[]> {
    const keys = texts.map((t) => createHash("sha1").update(t).digest("hex"));
    const missing: number[] = [];
    for (let i = 0; i < texts.length; i++) if (!this.map.has(keys[i]!)) missing.push(i);
    if (missing.length > 0) {
      const vecs = await this.provider.embed(missing.map((i) => texts[i]!));
      missing.forEach((i, j) => this.map.set(keys[i]!, vecs[j]!));
    }
    return keys.map((k) => this.map.get(k)!);
  }
}

/* ------------------------------- file plumbing ------------------------------- */

async function collectFiles(
  root: string,
  cfg: Config,
  includeExtensions: ReadonlySet<string>,
  extraExcludes: string[],
): Promise<{ files: string[]; truncated: boolean }> {
  const excluded = new Set([...DEFAULT_EXCLUDED_DIRS_SET, ...cfg.extraExcludedDirs, ...extraExcludes]);
  const out: string[] = [];
  // Set when the per-call cap forced an early return; every such return leaves
  // remaining entries unvisited, so the flag is always honest (never spuriously true).
  let truncated = false;
  const atCap = () => out.length >= cfg.maxFilesPerCall;
  const walk = async (dir: string): Promise<void> => {
    if (atCap()) {
      truncated = true;
      return;
    }
    const entries = await fs.readdir(dir, { withFileTypes: true }).catch(() => []);
    entries.sort((a, b) => a.name.localeCompare(b.name));
    for (const e of entries) {
      if (atCap()) {
        truncated = true;
        return;
      }
      const p = `${dir}${sep}${e.name}`;
      if (e.isDirectory()) {
        if (e.name.startsWith(".") || excluded.has(e.name.toLowerCase())) continue;
        await walk(p);
      } else if (e.isFile()) {
        if (e.name.startsWith(".")) continue;
        if (isSupportedPath(p, includeExtensions)) out.push(p);
      }
    }
  };
  await walk(root);
  return { files: out, truncated };
}

export function isSupportedPath(path: string, includeExtensions: ReadonlySet<string>): boolean {
  const base = path.split("/").pop()!.toLowerCase();
  const ext = base.slice(base.lastIndexOf("."));
  if (includeExtensions.has(ext)) return true;
  // extensionless special names, e.g. Dockerfile / Makefile
  if (!base.includes(".")) {
    return ["dockerfile", "makefile", "justfile"].includes(base);
  }
  return false;
}

function extensionIncludeSet(extra?: string[]): Set<string> {
  const set = new Set(defaultExtensions());
  for (const e of extra ?? []) set.add(e.startsWith(".") ? e.toLowerCase() : `.${e.toLowerCase()}`);
  return set;
}

function clampInt(v: number | undefined, min: number, max: number, fallback: number): number {
  if (v === undefined || !Number.isFinite(v)) return fallback;
  return Math.max(min, Math.min(max, Math.floor(v)));
}

interface FileIndexOutcome {
  status: "added" | "skipped" | "error";
  chunks?: number;
  error?: string;
}

/** Index (or skip) one file into the store; never throws for per-file issues. */
async function indexSingleFile(
  store: VectorStore,
  path: string,
  embedder: { embed(texts: string[]): Promise<Float32Array[]> },
  identity: StoreIdentity,
  opts: {
    maxFileBytes: number;
    chunkSize: number;
    chunkOverlap: number;
    force: boolean;
    cache?: EmbedCache;
  },
): Promise<FileIndexOutcome> {
  try {
    const kind = fileKindFor(path);
    if (!kind) return { status: "error", error: "unsupported file type (add ext via `include`)" };
    const buf = await fs.readFile(path);
    const hash = createHash("sha256").update(buf).digest("hex");
    if (!opts.force && store.getFile(path)?.contentHash === hash) return { status: "skipped" };
    const stat = await fs.stat(path);
    const extracted = await extractBytes(path, buf, kind, opts.maxFileBytes);
    const rawChunks = chunkExtracted(kind, extracted, {
      chunkSize: opts.chunkSize,
      chunkOverlap: opts.chunkOverlap,
      maxChunksPerFile: 4000,
    });
    if (rawChunks.length === 0) return { status: "error", error: "no extractable text chunks" };
    const cache = opts.cache ?? new EmbedCache(embedder);
    const rows = await cache.embedAll(rawChunks.map((c) => c.text));
    const dim = rows[0]!.length;
    store.add(
      {
        path,
        kind,
        sizeBytes: stat.size,
        mtimeMs: stat.mtimeMs,
        contentHash: hash,
        indexedAt: new Date().toISOString(),
      },
      rawChunks.map((c) => ({
        fileId: 0, // re-bound store-side
        text: c.text,
        title: c.title,
        startLine: c.startLine,
        endLine: c.endLine,
        page: c.page,
      })),
      Float32Array.from(rows.flatMap((r) => Array.from(r))),
      dim,
      identity,
    );
    return { status: "added", chunks: rawChunks.length };
  } catch (err) {
    if (err instanceof SmartDocsError) return { status: "error", error: err.toStringWithHint() };
    const msg = err instanceof Error ? err.message : String(err);
    return { status: "error", error: msg };
  }
}

/* --------------------------------- tool defs --------------------------------- */

export interface ToolDef {
  name: string;
  title: string;
  description: string;
  inputSchema: z.ZodRawShape;
  annotations?: {
    readOnlyHint?: boolean;
    destructiveHint?: boolean;
    idempotentHint?: boolean;
    openWorldHint?: boolean;
  };
  handler: (args: Record<string, unknown>, extra: Extra) => Promise<CallToolResult>;
}

const KIND_ENUM = ["markdown", "text", "rst", "pdf", "code"] as const;

function hitToOutput(h: SearchHit): Record<string, unknown> {
  return {
    path: h.path,
    ...(h.title ? { section: h.title } : {}),
    ...(h.page !== undefined ? { page: h.page } : { lines: `${h.startLine}-${h.endLine}` }),
    score: Math.round(h.score * 1000) / 1000,
    text: h.text,
  };
}

/** Build the six tools bound to concrete services. */
export function defineTools(services: Services): ToolDef[] {
  const cfg: Config = services.cfg;
  const embedder = services.createEmbedder();
  const identity: StoreIdentity = services.identity(embedder);

  const requireIndexed = async (): Promise<VectorStore> => {
    const store = await services.store();
    store.assertCompatible(identity);
    if (store.count === 0) {
      throw new EmptyIndexError(
        "The index is empty — nothing has been indexed yet.",
        "Call index_directory first, e.g. { \"path\": \"/abs/path/to/docs\" }.",
      );
    }
    return store;
  };

  const toPassages = (hits: SearchHit[]): SummaryPassage[] =>
    hits.map((h) => ({
      text: h.text,
      path: h.path,
      startLine: h.startLine,
      endLine: h.endLine,
      page: h.page,
      title: h.title,
    }));

  return [
    {
      name: "index_directory",
      title: "Index a documentation folder",
      description:
        "Recursively scan a folder and build/refresh the offline semantic index. Extracts text from .md, .mdx, .txt, .rst, .pdf and common code files, chunks it intelligently (markdown/RST/heading-aware), embeds and stores vectors locally. Incremental: unchanged files are skipped by content hash. Sends progress notifications; safe to re-run after edits.",
      inputSchema: {
        path: z.string().describe("Directory to index (absolute path recommended; ~ is expanded)."),
        include: z
          .array(z.string())
          .optional()
          .describe(
            'Extra file extensions to include, e.g. [".adoc", ".tex"]. Defaults already cover .md/.mdx/.txt/.rst/.pdf plus common code files.',
          ),
        force: z.boolean().optional().describe("Re-embed every file even if its content hash matches the stored index."),
        chunkSize: z
          .number()
          .int()
          .positive()
          .optional()
          .describe("Target chunk size in characters (default from server config, usually 1200)."),
        chunkOverlap: z
          .number()
          .int()
          .nonnegative()
          .optional()
          .describe("Overlap between consecutive chunks in characters (default from server config, usually 200; must be < chunkSize)."),
        excludeDirs: z
          .array(z.string())
          .optional()
          .describe("Extra directory names to skip (built-ins like node_modules/.git/dist are always skipped)."),
      },
      annotations: { readOnlyHint: false, destructiveHint: false, idempotentHint: true, openWorldHint: false },
      handler: async (raw, extra) => {
        try {
          const args = raw as unknown as {
            path: string;
            include?: string[];
            force?: boolean;
            chunkSize?: number;
            chunkOverlap?: number;
            excludeDirs?: string[];
          };
          const root = resolve(args.path.replace(/^~(?=\/|$)/, process.env.HOME ?? "."));
          const stat = await fs.stat(root).catch(() => null);
          if (!stat) {
            throw new SmartDocsError(
              `Directory not found: ${args.path}`,
              "Pass an existing directory. Use list_indexed to review what is already indexed.",
            );
          }
          if (!stat.isDirectory()) {
            throw new SmartDocsError(
              `${args.path} is not a directory.`,
              "For single files use get_file_summary; index_directory indexes a whole folder.",
            );
          }

          const store = await services.store();
          const chunkSize = clampInt(args.chunkSize, 100, 100_000, cfg.chunkSize);
          const chunkOverlap = clampInt(args.chunkOverlap, 0, chunkSize - 1, Math.min(cfg.chunkOverlap, chunkSize - 1));

          const { files, truncated } = await collectFiles(root, cfg, extensionIncludeSet(args.include), args.excludeDirs ?? []);
          if (files.length === 0) {
            throw new SmartDocsError(
              `No indexable files found under ${root}.`,
              "The folder may be empty or contain only unsupported types. Pass `include` to add extensions (README § Extending file types).",
            );
          }

          await reportLog(extra, "info", `smart-docs: indexing ${files.length} file(s) under ${root}`);
          const cache = new EmbedCache(embedder);
          let added = 0;
          let skipped = 0;
          let newChunks = 0;
          const errors: Array<{ file: string; error: string }> = [];
          const started = Date.now();
          let sinceSave = 0;

          for (let i = 0; i < files.length; i++) {
            const file = files[i]!;
            await reportProgress(extra, { done: i, total: files.length, currentFile: file });
            const outcome = await indexSingleFile(store, file, embedder, identity, {
              maxFileBytes: cfg.maxFileBytes,
              chunkSize,
              chunkOverlap,
              force: args.force === true,
              cache,
            });
            if (outcome.status === "added") {
              added++;
              newChunks += outcome.chunks ?? 0;
              if (++sinceSave >= 25) {
                sinceSave = 0;
                await store.save(); // checkpoint long runs
              }
            } else if (outcome.status === "skipped") {
              skipped++;
            } else {
              errors.push({ file, error: outcome.error ?? "unknown" });
            }
          }

          // Drop files under this root that no longer exist on disk.
          let removed = 0;
          const prefix = root.endsWith(sep) ? root : root + sep;
          for (const path of store.indexedPaths()) {
            if (!path.startsWith(prefix)) continue;
            if (!(await fs.stat(path).then(() => true).catch(() => false))) {
              store.removeFile(path);
              removed++;
            }
          }

          await store.save();
          await reportProgress(extra, { done: files.length, total: files.length });
          return jsonResult({
            ok: true,
            summary:
              `Indexed ${added} file(s), ${newChunks} new chunk(s); skipped ${skipped} unchanged; ` +
              `removed ${removed} stale; ${errors.length} error(s) in ${((Date.now() - started) / 1000).toFixed(1)}s.` +
              (truncated
                ? ` Stopped early at the ${cfg.maxFilesPerCall}-file cap: some files under ${root} were NOT indexed.`
                : ""),
            root,
            counts: { added, skipped, removed, failed: errors.length, discovered: files.length },
            ...(truncated
              ? {
                  filesTruncated: true,
                  truncationHint: `The per-call file cap (${cfg.maxFilesPerCall}) stopped discovery early. Raise SMART_DOCS_MAX_FILES, or index subfolders in multiple calls.`,
                }
              : {}),
            indexStats: store.stats(),
            ...(errors.length > 0
              ? {
                  errors: errors.slice(0, 20),
                  errorHint:
                    "Per-file failures do not stop indexing. Common fixes: raise SMART_DOCS_MAX_FILE_MB, or exclude the path.",
                }
              : {}),
          });
        } catch (err) {
          return errorResult(err);
        }
      },
    },
    {
      name: "semantic_search",
      title: "Semantic search across indexed docs",
      description:
        "Search all indexed documentation and code with natural language (cosine similarity over local embeddings). Returns the most relevant chunks with file path, section/lines or page, similarity score, chunk text, and surrounding context when requested. Fully offline after indexing.",
      inputSchema: {
        query: z.string().describe("Natural-language query, e.g. 'how are payment retries handled?'"),
        k: z.number().int().positive().optional().describe("Number of results (server default, usually 8)."),
        paths: z.array(z.string()).optional().describe("Restrict to file path prefixes, e.g. ['docs/api', 'guides']."),
        extensions: z.array(z.string()).optional().describe("Restrict to extensions, e.g. ['.md', '.rst']."),
        kinds: z.array(z.enum(KIND_ENUM)).optional().describe("Restrict to doc kinds."),
      },
      annotations: { readOnlyHint: true, openWorldHint: false },
      handler: async (raw) => {
        try {
          const args = raw as unknown as {
            query: string;
            k?: number;
            paths?: string[];
            extensions?: string[];
            kinds?: FileKind[];
          };
          const store = await requireIndexed();
          const k = clampInt(args.k, 1, 50, cfg.defaultTopK);
          const [qv] = await embedder.embed([args.query]);
          if (!qv) throw new SmartDocsError("Failed to embed the query.", "Check embedding provider logs; retry.");
          // Paths given as partial segments should behave like prefixes from any root.
          const normPaths = args.paths?.map((p) => `/${p.replace(/^\//, "").replace(/\/+$/, "")}/`);
          const hits = store.search(qv, k, {
            paths: normPaths,
            extensions: args.extensions?.map((e) => (e.startsWith(".") ? e.toLowerCase() : `.${e.toLowerCase()}`)),
            kinds: args.kinds,
          });
          return jsonResult({
            query: args.query,
            results: hits.map(hitToOutput),
            ...(hits.length === 0
              ? { hint: "No matches. Try different wording (synonyms), lower the thresholds by using more folders, or re-index." }
              : {}),
          });
        } catch (err) {
          return errorResult(err);
        }
      },
    },
    {
      name: "ask_docs",
      title: "Answer a question from indexed docs only",
      description:
        "Answer a question using ONLY the indexed knowledge with citations. Retrieval finds the top passages; with an LLM configured (SMART_DOCS_LLM_*), produces a synthesized answer with [n] citations; otherwise returns a structured extractive answer of the passages themselves. Use semantic_search when you need raw chunks instead of an answer.",
      inputSchema: {
        question: z
          .string()
          .describe("The question, e.g. 'what is the retry policy for webhooks and where is it documented?'"),
        k: z.number().int().positive().optional().describe("Number of passages to ground the answer (default matches server config)."),
      },
      annotations: { readOnlyHint: true, openWorldHint: false },
      handler: async (raw) => {
        try {
          const args = raw as unknown as { question: string; k?: number };
          const store = await requireIndexed();
          const k = clampInt(args.k, 1, 50, cfg.defaultTopK);
          const [qv] = await embedder.embed([args.question]);
          if (!qv) throw new SmartDocsError("Failed to embed the question.", "Check embedding provider logs; retry.");
          const hits = store.search(qv, k);
          const passages = toPassages(hits);
          const llm = services.llm();
          let answer: string;
          let mode: "llm" | "extractive";
          if (llm && passages.length > 0) {
            answer = await askWithLlm(llm, args.question, passages);
            mode = "llm";
          } else {
            mode = "extractive";
            const picked = extractiveSummary(
              hits.map((h) => ({ meta: h, path: h.path })),
              10,
            );
            answer =
              passages.length === 0
                ? "No relevant passages found for this question."
                : "Extractive answer (offline; set SMART_DOCS_LLM_* for a synthesized answer):\n" +
                  picked
                    .map((p) => {
                      const loc = p.page !== undefined ? `page ${p.page}` : `lines ${p.startLine}-${p.endLine}`;
                      return `• ${p.text}  [${p.path} ${loc}]`;
                    })
                    .join("\n");
          }
          return jsonResult({
            question: args.question,
            mode,
            answer,
            sources: clipTotalPassages(passages).map((p) => ({
              path: p.path,
              ...(p.title ? { section: p.title } : {}),
              ...(p.page !== undefined ? { page: p.page } : { lines: `${p.startLine}-${p.endLine}` }),
            })),
            ...(passages.length === 0
              ? { hint: "Nothing matched. Try semantic_search with different words, or index more folders." }
              : {}),
          });
        } catch (err) {
          return errorResult(err);
        }
      },
    },
    {
      name: "get_file_summary",
      title: "Summarize a file or directory",
      description:
        "Summarize one file (all of its chunks) or a whole directory (lead chunks of each file under it). Unindexed single files are indexed automatically first. Uses the configured LLM when available, otherwise offline extractive summarization with citations.",
      inputSchema: {
        path: z.string().describe("Absolute path of the file or directory to summarize."),
      },
      annotations: { readOnlyHint: true, openWorldHint: false },
      handler: async (raw, extra) => {
        try {
          const args = raw as unknown as { path: string };
          const target = resolve(args.path.replace(/^~(?=\/|$)/, process.env.HOME ?? "."));
          const stat = await fs.stat(target).catch(() => null);
          if (!stat) throw new SmartDocsError(`No such file: ${args.path}`, "Pass an absolute path that exists on disk.");

          let label: string;
          let passages: SummaryPassage[];
          if (stat.isDirectory()) {
            const store = await services.store();
            const prefix = target.endsWith(sep) ? target : target + sep;
            const files = store.indexedPaths().filter((p) => p.startsWith(prefix));
            if (files.length === 0) {
              throw new SmartDocsError(
                `Nothing indexed under ${args.path}.`,
                "Run index_directory with this folder first.",
              );
            }
            passages = [];
            for (const f of files.slice(0, 60)) {
              for (const c of store.chunksOf(f).slice(0, 3)) {
                passages.push({ text: c.text, path: f, startLine: c.startLine, endLine: c.endLine, page: c.page, title: c.title });
              }
            }
            label = "directory";
          } else {
            const store = await services.store();
            store.assertCompatible(identity);
            if (!store.getFile(target)) {
              await reportLog(extra, "info", `smart-docs: ${target} is not indexed yet — indexing it now`);
              const outcome = await indexSingleFile(store, target, embedder, identity, {
                maxFileBytes: cfg.maxFileBytes,
                chunkSize: cfg.chunkSize,
                chunkOverlap: cfg.chunkOverlap,
                force: false,
              });
              await store.save();
              if (outcome.status !== "added") {
                throw new SmartDocsError(
                  `Could not index ${args.path}: ${outcome.error ?? outcome.status}`,
                  "See README for supported file types; index_directory reports detailed per-file errors.",
                );
              }
            }
            const chunks = store.chunksOf(target);
            if (chunks.length === 0) {
              throw new SmartDocsError(
                `No text chunks available to summarize for ${args.path}.`,
                "The file may be empty or unsupported; index_directory reports per-file status.",
              );
            }
            passages = chunks.map((c): SummaryPassage => ({
              text: c.text,
              path: target,
              startLine: c.startLine,
              endLine: c.endLine,
              page: c.page,
              title: c.title,
            }));
            label = "file";
          }

          passages = clipTotalPassages(passages);
          if (passages.length === 0) {
            throw new SmartDocsError(
              `No text chunks available to summarize for ${args.path}.`,
              "The file may be empty or unsupported; index_directory reports per-file status.",
            );
          }
          const llm = services.llm();
          let summary: string;
          let mode: "llm" | "extractive";
          if (llm) {
            summary = await summarizeWithLlm(llm, label, passages);
            mode = "llm";
          } else {
            mode = "extractive";
            const picked = extractiveSummary(
              passages.map((p) => ({ meta: { text: p.text, startLine: p.startLine, endLine: p.endLine, page: p.page, title: p.title, fileId: 0 }, path: p.path })),
              10,
            );
            summary =
              "Extractive summary (offline; set SMART_DOCS_LLM_* for synthesized summaries):\n" +
              picked
                .map((p) => {
                  const loc = p.page !== undefined ? `page ${p.page}` : `lines ${p.startLine}-${p.endLine}`;
                  return `• ${p.text}  [${p.path} ${loc}]`;
                })
                .join("\n");
          }
          return jsonResult({
            path: target,
            kind: label,
            mode,
            summary,
            sources: passages.slice(0, 12).map((p) => ({
              path: p.path,
              ...(p.title ? { section: p.title } : {}),
              ...(p.page !== undefined ? { page: p.page } : { lines: `${p.startLine}-${p.endLine}` }),
            })),
          });
        } catch (err) {
          return errorResult(err);
        }
      },
    },
    {
      name: "list_indexed",
      title: "Show indexed files and stats",
      description:
        "Show what smart-docs has indexed: embedding model/provider/dimension, chunk and file counts, per-kind breakdown, and the full file list with chunk counts and timestamps.",
      inputSchema: {},
      annotations: { readOnlyHint: true, openWorldHint: false },
      handler: async () => {
        try {
          const store = await services.store();
          const stats = store.stats();
          const files = store.listFiles().sort((a, b) => a.path.localeCompare(b.path));
          const byKind = files.reduce<Record<string, number>>((acc, f) => {
            acc[f.kind] = (acc[f.kind] ?? 0) + 1;
            return acc;
          }, {});
          return jsonResult({
            indexDir: services.indexDir,
            model: stats.model,
            provider: stats.provider,
            dim: stats.dim,
            chunkCount: stats.chunkCount,
            fileCount: stats.fileCount,
            byKind,
            files: files.map((f) => ({
              path: f.path,
              kind: f.kind,
              chunks: f.chunkCount,
              sizeBytes: f.sizeBytes,
              indexedAt: f.indexedAt,
            })),
            ...(stats.fileCount === 0
              ? { hint: "Nothing indexed yet — call index_directory with a folder path." }
              : {}),
          });
        } catch (err) {
          return errorResult(err);
        }
      },
    },
    {
      name: "clear_index",
      title: "Clear the index",
      description:
        "Delete the persisted vector index (chunk metadata + embeddings). Safe by default: requires confirm: true, and the local model cache is kept. Use when switching embedding models or starting fresh, then call index_directory again.",
      inputSchema: {
        confirm: z
          .boolean()
          .optional()
          .describe("Safety guard: when true (and only true) the index is deleted; omitted/false is refused with instructions."),
      },
      annotations: { readOnlyHint: false, destructiveHint: true, idempotentHint: true, openWorldHint: false },
      handler: async (raw) => {
        try {
          const args = raw as unknown as { confirm?: boolean };
          if (args.confirm !== true) {
            throw new SmartDocsError(
              "clear_index requires confirm: true.",
              "Re-call the tool with { \"confirm\": true } to actually delete. Call list_indexed if you only meant to inspect.",
            );
          }
          await VectorStore.clearOnDisk(services.indexDir);
          const store = await services.reloadStore();
          return jsonResult({
            cleared: true,
            indexDir: services.indexDir,
            chunkCount: store.count,
            hint: "Index is empty now; call index_directory to rebuild.",
          });
        } catch (err) {
          return errorResult(err);
        }
      },
    },
  ];
}

/* --------------------------------- constants ---------------------------------- */

const DEFAULT_EXCLUDED_DIRS_SET = new Set(
  [
    "node_modules", ".git", ".hg", ".svn",
    ".smart-docs", ".next", ".nuxt", ".turbo", ".cache",
    "dist", "build", "out", "coverage", "target",
    "venv", ".venv", "__pycache__", ".idea", ".vscode",
  ].map((d) => d.toLowerCase()),
);
