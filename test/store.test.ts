import { describe, expect, it, afterEach } from "vitest";
import { promises as fs } from "node:fs";
import { join, resolve } from "node:path";
import { VectorStore } from "../src/store.js";
import type { ChunkMeta } from "../src/types.js";

const dirs: string[] = [];
function tmpDir(name: string): string {
  const dir = join(resolve("test/.tmp"), name);
  dirs.push(dir);
  return dir;
}
afterEach(async () => {
  for (const dir of dirs) await fs.rm(dir, { recursive: true, force: true });
  dirs.length = 0;
});

const identity = { model: "test-model", provider: "test" };

function fileRec(path: string, kind = "markdown" as const) {
  return { path, kind, sizeBytes: 10, mtimeMs: 1, contentHash: `hash-${path}`, indexedAt: "now" };
}

function chunksFor(path: string, n: number, dim: number): ChunkMeta[] {
  return Array.from({ length: n }, (_, i) => ({
    fileId: 0,
    text: `chunk of ${path} #${i}`,
    startLine: i + 1,
    endLine: i + 1,
  }));
}

function rowsFor(count: number, dim: number): Float32Array {
  const rows = new Float32Array(count * dim);
  // deterministic pseudo-vectors: unit vector on axis (i % dim)
  for (let i = 0; i < count; i++) rows[i * dim + (i % dim)] = 1;
  return rows;
}

describe("VectorStore", () => {
  it("adds, searches by cosine, and returns file metadata", async () => {
    const store = await VectorStore.load(tmpDir("search"));
    const dim = 4;
    store.add(fileRec("/a.md"), chunksFor("/a.md", 3, dim), rowsFor(3, dim), dim, identity);
    store.add(fileRec("/b.md", "text"), chunksFor("/b.md", 3, dim), rowsFor(3, dim), dim, identity);
    expect(store.count).toBe(6);
    expect(store.fileCount).toBe(2);

    const hits = store.search(Float32Array.from([1, 0, 0, 0]), 2);
    expect(hits).toHaveLength(2);
    expect(hits[0]!.score).toBeCloseTo(1, 5);
    expect(hits[0]!.path).toBe("/a.md"); // chunk 0 of a.md is axis-0
    expect(hits[1]!.score).toBeGreaterThanOrEqual(0);
  });

  it("filters by extension and kind", async () => {
    const store = await VectorStore.load(tmpDir("filter"));
    const dim = 4;
    store.add(fileRec("/a.md"), chunksFor("/a.md", 2, dim), rowsFor(2, dim), dim, identity);
    store.add(fileRec("/b.ts", "code"), chunksFor("/b.ts", 2, dim), rowsFor(2, dim), dim, identity);
    store.add(fileRec("/c/list.txt", "text"), chunksFor("/c/list.txt", 2, dim), rowsFor(2, dim), dim, identity);

    const q = Float32Array.from([1, 0, 0, 0]);
    const txtHits = store.search(q, 10, { extensions: [".txt"] });
    expect(txtHits.length).toBeGreaterThan(0);
    expect(txtHits.every((h) => h.path === "/c/list.txt")).toBe(true);

    const codeHits = store.search(q, 10, { kinds: ["code"] });
    expect(codeHits.length).toBeGreaterThan(0);
    expect(codeHits.every((h) => h.path === "/b.ts")).toBe(true);

    const pathHits = store.search(q, 10, { paths: ["/c/"] });
    expect(pathHits.length).toBeGreaterThan(0);
    expect(pathHits.every((h) => h.path === "/c/list.txt")).toBe(true);

    const mdHits = store.search(q, 10, { extensions: [".md"] });
    expect(mdHits.every((h) => h.path === "/a.md")).toBe(true);

    const all = store.search(q, 10);
    expect(all.length).toBeGreaterThan(txtHits.length + mdHits.length); // filters narrowed the set
  });

  it("removeFile rebuilds vectors and keeps search correct", async () => {
    const store = await VectorStore.load(tmpDir("remove"));
    const dim = 4;
    store.add(fileRec("/a.md"), chunksFor("/a.md", 3, dim), rowsFor(3, dim), dim, identity);
    store.add(fileRec("/b.md"), chunksFor("/b.md", 3, dim), rowsFor(3, dim), dim, identity);
    expect(store.removeFile("/a.md")).toBe(true);
    expect(store.count).toBe(3);
    const hits = store.search(Float32Array.from([0, 1, 0, 0]), 10);
    expect(hits).toHaveLength(3);
    expect(hits[0]!.path).toBe("/b.md");
    expect(store.removeFile("/missing.md")).toBe(false);
  });

  it("persists and reloads with intact search results + model identity", async () => {
    const dir = tmpDir("persist");
    const s1 = await VectorStore.load(dir);
    const dim = 4;
    s1.add(fileRec("/a.md"), chunksFor("/a.md", 4, dim), rowsFor(4, dim), dim, { model: "m1", provider: "p1" });
    await s1.save();

    const s2 = await VectorStore.load(dir);
    expect(s2.count).toBe(4);
    expect(s2.model).toBe("m1");
    expect(s2.provider).toBe("p1");
    expect(s2.dim).toBe(dim);
    const hits = s2.search(Float32Array.from([0, 0, 1, 0]), 10);
    expect(hits[0]!.score).toBeCloseTo(1, 5);
    expect(hits[0]!.text).toContain("chunk of /a.md");
  });

  it("rejects mixing models with a recoverable error", async () => {
    const store = await VectorStore.load(tmpDir("mismatch"));
    const dim = 4;
    store.add(fileRec("/a.md"), chunksFor("/a.md", 1, dim), rowsFor(1, dim), dim, { model: "m1", provider: "p1" });
    expect(() => store.add(fileRec("/b.md"), chunksFor("/b.md", 1, dim), rowsFor(1, dim), dim, { model: "m2", provider: "p1" }))
      .toThrow(/built with provider "p1" model "m1"/);
    expect(() => store.search(Float32Array.from([1, 0, 0, 0.1]), 1)).not.toThrow();
    // queries with wrong dim throw a smart error
    expect(() => store.search(Float32Array.from([1, 0, 0]), 1)).toThrow(/dimensions but the index/);
  });

  it("flags corrupt index files instead of crashing later", async () => {
    const dir = tmpDir("corrupt");
    const s1 = await VectorStore.load(dir);
    s1.add(fileRec("/a.md"), chunksFor("/a.md", 2, 4), rowsFor(2, 4), 4, identity);
    await s1.save();
    await fs.writeFile(join(dir, "vectors.bin"), "nonsense");
    await expect(VectorStore.load(dir)).rejects.toThrow(/corrupt/i);
  });

  it("clearOnDisk resets to a fresh store", async () => {
    const dir = tmpDir("clear");
    const s1 = await VectorStore.load(dir);
    s1.add(fileRec("/a.md"), chunksFor("/a.md", 2, 4), rowsFor(2, 4), 4, identity);
    await s1.save();
    await VectorStore.clearOnDisk(dir);
    const s2 = await VectorStore.load(dir);
    expect(s2.count).toBe(0);
    expect(s2.fileCount).toBe(0);
    expect(s2.model).toBe("");
  });
});
