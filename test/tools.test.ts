import { describe, expect, it, beforeEach, afterEach } from "vitest";
import { promises as fs } from "node:fs";
import { join, resolve } from "node:path";
import { loadConfig, type Config } from "../src/config.js";
import { Services } from "../src/services.js";
import type { EmbeddingProvider } from "../src/embeddings.js";
import { defineTools } from "../src/tools.js";
import { VectorStore } from "../src/store.js";
import type { CallToolResult } from "@modelcontextprotocol/sdk/types.js";

const root = resolve("test/.tmp/tools-fixture");
const dirs = [root, resolve("test/.tmp/tools-data")];

/**
 * Deterministic 8-dim "hash" embedder: bag-of-token hashing so that
 * semantically-identical texts map to identical vectors and searches are
 * reproducible offline.
 */
class StubEmbedder implements EmbeddingProvider {
  readonly provider = "stub";
  readonly model = "stub-8dim";
  async embed(texts: string[]): Promise<Float32Array[]> {
    return texts.map((t) => {
      const v = new Float32Array(8);
      for (const tok of t.toLowerCase().split(/\W+/).filter(Boolean)) {
        let h = 0;
        for (let i = 0; i < tok.length; i++) h = (h * 31 + tok.charCodeAt(i)) >>> 0;
        v[h % 8] += 1;
      }
      const n = Math.sqrt(v.reduce((s, x) => s + x * x, 0)) || 1;
      for (let i = 0; i < 8; i++) v[i] = v[i]! / n;
      return v;
    });
  }
}

function makeServices(overrides: Partial<Config> = {}): Services {
  const base = loadConfig([], {});
  const cfg: Config = {
    ...base,
    dataRoot: resolve("test/.tmp/tools-data"),
    maxFilesPerCall: 100,
    defaultTopK: 5,
    ...overrides,
  };
  const services = new Services(cfg);
  // stub out embedding creation for offline, fast, deterministic tests
  (services as unknown as { createEmbedder(): EmbeddingProvider }).createEmbedder = () => new StubEmbedder();
  return services;
}

/** Simulate an MCP client: call a tool excluding the SDK plumbing. */
async function call(
  tools: ReturnType<typeof defineTools>,
  name: string,
  args: Record<string, unknown> = {},
): Promise<CallToolResult & { parsed: unknown }> {
  const tool = tools.find((t) => t.name === name)!;
  const result = await tool.handler(args, {
    requestId: 1,
    sendNotification: async () => undefined,
  } as unknown as Parameters<typeof tool.handler>[1]);
  return { ...result, parsed: JSON.parse(result.content[0]!.type === "text" ? result.content[0]!.text : "{}") };
}

beforeEach(async () => {
  for (const d of dirs) await fs.rm(d, { recursive: true, force: true });
  await fs.mkdir(join(root, "guides"), { recursive: true });
  await fs.writeFile(
    join(root, "payments.md"),
    [
      "# Payments",
      "",
      "We bill monthly. Refunds are processed within 5 business days.",
      "",
      "## Webhook retries",
      "",
      "Failed webhook deliveries retry with exponential backoff up to 5 times over one hour.",
    ].join("\n"),
    "utf8",
  );
  await fs.writeFile(join(root, "guides"), "", { flag: "wx" }).catch(() => undefined);
  await fs.writeFile(
    join(root, "guides", "shipping.txt"),
    "Free shipping applies to orders over $50.\nExpedited shipping costs $12.",
    "utf8",
  );
  await fs.writeFile(join(root, "notes.xyz"), "unsupported format");
  await fs.mkdir(join(root, "node_modules"), { recursive: true });
  await fs.writeFile(join(root, "node_modules", "dep.md"), "# Should be ignored", "utf8");
});

afterEach(async () => {
  for (const d of dirs) await fs.rm(d, { recursive: true, force: true });
  await VectorStore.clearOnDisk(join(resolve("test/.tmp/tools-data"), "index"));
});

describe("smart-docs tools (integration)", () => {
  it("index_directory indexes supported files, skips unsupported and excluded dirs, then reports stats", async () => {
    const services = makeServices();
    const tools = defineTools(services);
    const res = await call(tools, "index_directory", { path: root });
    const parsed = res.parsed as Record<string, never> & { counts?: Record<string, number>; ok?: boolean };
    void parsed;
    expect(res.isError).toBeUndefined();
    const payload = res.parsed as { ok: boolean; counts: { added: number; failed: number }; indexStats: { fileCount: number } };
    expect(payload.ok).toBe(true);
    expect(payload.counts.added).toBeGreaterThanOrEqual(2); // payments.md + shipping.txt
    expect(payload.indexStats.fileCount).toBe(payload.counts.added);
  });

  it("is incremental: re-running skips unchanged files", async () => {
    const services = makeServices();
    const tools = defineTools(services);
    const first = (await call(tools, "index_directory", { path: root })).parsed as { counts: { added: number; skipped: number } };
    expect(first.counts.added).toBeGreaterThanOrEqual(2);
    const second = (await call(tools, "index_directory", { path: root })).parsed as { counts: { added: number; skipped: number } };
    expect(second.counts.added).toBe(0);
    expect(second.counts.skipped).toBeGreaterThanOrEqual(2);
  });

  it("list_indexed reflects index state", async () => {
    const services = makeServices();
    const tools = defineTools(services);
    await call(tools, "index_directory", { path: root });
    const listing = (await call(tools, "list_indexed")).parsed as { model: string; chunkCount: number; fileCount: number; files: Array<{ path: string }> };
    expect(listing.fileCount).toBeGreaterThanOrEqual(2);
    expect(listing.model).toBe("stub-8dim");
    expect(listing.files.some((f) => f.path.endsWith("payments.md"))).toBe(true);
    expect(listing.files.some((f) => f.path.includes("unsupported-format-guard"))).toBe(false);
  });

  it("semantic_search returns sorted relevant chunks with locations", async () => {
    const services = makeServices();
    const tools = defineTools(services);
    await call(tools, "index_directory", { path: root });
    const res = await call(tools, "semantic_search", { query: "webhook retries exponential backoff" });
    expect(res.isError).toBeUndefined();
    const hits = (res.parsed as { results: Array<{ path: string; lines: string; score: number; text: string }> }).results;
    expect(hits.length).toBeGreaterThan(0);
    const top = hits[0]!;
    expect(top.path.endsWith("payments.md")).toBe(true);
    expect(top.score).toBeGreaterThanOrEqual(hits[hits.length - 1]!.score);
    expect(top.lines).toMatch(/^\d+-\d+$/);
  });

  it("semantic_search honors extension filters", async () => {
    const services = makeServices();
    const tools = defineTools(services);
    await call(tools, "index_directory", { path: root });
    const query = "webhook retries exponential backoff";
    const md = await call(tools, "semantic_search", { query, extensions: [".md"], k: 10 });
    const mdHits = (md.parsed as { results: Array<{ path: string }> }).results;
    expect(mdHits.length).toBeGreaterThan(0);
    expect(mdHits.every((h) => h.path.endsWith("payments.md"))).toBe(true);

    const txt = await call(tools, "semantic_search", { query: "free shipping", extensions: [".txt"], k: 10 });
    const txtHits = (txt.parsed as { results: Array<{ path: string }> }).results;
    expect(txtHits.every((h) => h.path.endsWith("shipping.txt"))).toBe(true);
    expect(txtHits.some((h) => h.score > 0.5)).toBe(true);
  });

  it("search/ask refuse with an actionable error when nothing is indexed", async () => {
    const services = makeServices();
    const tools = defineTools(services);
    const res = await call(tools, "semantic_search", { query: "anything" });
    expect(res.isError).toBe(true);
    const err = res.parsed as { error: string; hint: string };
    expect(err.error).toBe("EmptyIndexError");
    expect(err.hint).toContain("index_directory");
  });

  it("ask_docs returns cited extractive answer offline", async () => {
    const services = makeServices();
    const tools = defineTools(services);
    await call(tools, "index_directory", { path: root });
    const res = await call(tools, "ask_docs", { question: "how do webhook retries work?" });
    expect(res.isError).toBeUndefined();
    const payload = res.parsed as { mode: string; answer: string; sources: Array<{ path: string }> };
    expect(payload.mode).toBe("extractive");
    expect(payload.answer.length).toBeGreaterThan(0);
    expect(payload.sources.some((s) => s.path.endsWith("payments.md"))).toBe(true);
  });

  it("get_file_summary summarizes a single file and auto-indexes files that are missing", async () => {
    const services = makeServices();
    const tools = defineTools(services);
    await call(tools, "index_directory", { path: root });
    const payments = join(root, "payments.md");
    // force fresh store to exercise auto-index path
    await VectorStore.clearOnDisk(join(resolve("test/.tmp/tools-data"), "index"));
    await services.reloadStore();
    const res = await call(tools, "get_file_summary", { path: payments });
    expect(res.isError).toBeUndefined();
    const payload = res.parsed as { kind: string; mode: string; summary: string; sources: Array<{ path: string }> };
    expect(payload.kind).toBe("file");
    expect(payload.summary).toContain("bill monthly");
  });

  it("clear_index refuses without confirm, succeeds with it", async () => {
    const services = makeServices();
    const tools = defineTools(services);
    await call(tools, "index_directory", { path: root });
    const refused = await call(tools, "clear_index", {});
    expect(refused.isError).toBe(true);
    expect((refused.parsed as { hint: string }).hint).toContain("confirm");
    const ok = await call(tools, "clear_index", { confirm: true });
    expect(res(ok)).toBe(true);
    const listing = (await call(tools, "list_indexed")).parsed as { fileCount: number };
    expect(listing.fileCount).toBe(0);
  });
});

function res(r: CallToolResult & { parsed: unknown }): boolean {
  return (r.parsed as { cleared: boolean }).cleared === true;
}
