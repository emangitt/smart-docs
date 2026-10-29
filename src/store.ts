import { promises as fs } from "node:fs";
import { join } from "node:path";
import type {
  ChunkMeta,
  FileKind,
  FileRecord,
  PersistedStoreMeta,
  SearchFilter,
  SearchHit,
} from "./types.js";
import { ModelMismatchError, SmartDocsError, StoreCorruptError } from "./errors.js";

const META_FILE = "meta.json";
const CHUNKS_FILE = "chunks.json";
const VECTORS_FILE = "vectors.bin";

export interface StoreIdentity {
  model: string;
  provider: string;
}

/**
 * In-memory flat vector index with on-disk persistence. Chunk metadata is JSON;
 * embeddings are raw little-endian float32 rows in `vectors.bin`, one row per
 * chunk. Vectors are unit-normalized on insert, so search is a plain dot
 * product per chunk — fast enough for tens of thousands of chunks on a laptop
 * and, crucially, dependency-free.
 */
export class VectorStore {
  /** Identity of the model that produced the vectors; managed by the caller. */
  model = "";
  provider = "";
  dim = 0;

  private files = new Map<string, FileRecord>();
  private chunks: ChunkMeta[] = [];
  private vectors = new Float32Array(0);
  private nextFileId = 1;

  private constructor(public readonly dir: string) {}

  get fileCount(): number {
    return this.files.size;
  }

  get count(): number {
    return this.chunks.length;
  }

  listFiles(): FileRecord[] {
    return [...this.files.values()];
  }

  getFile(path: string): FileRecord | undefined {
    return this.files.get(path);
  }

  /** Absolute paths currently in the index. */
  indexedPaths(): string[] {
    return [...this.files.keys()];
  }

  /** Chunks belonging to one file, in document order. */
  chunksOf(path: string): ChunkMeta[] {
    const rec = this.files.get(path);
    if (!rec) return [];
    return this.chunks.filter((c) => c.fileId === rec.id);
  }

  static async load(dir: string): Promise<VectorStore> {
    const store = new VectorStore(dir);
    let metaRaw: string;
    try {
      metaRaw = await fs.readFile(join(dir, META_FILE), "utf8");
    } catch {
      return store; // fresh, empty store
    }
    try {
      const meta = JSON.parse(metaRaw) as PersistedStoreMeta;
      const chunks = JSON.parse(await fs.readFile(join(dir, CHUNKS_FILE), "utf8")) as ChunkMeta[];
      const vecBuf = await fs.readFile(join(dir, VECTORS_FILE));
      const vectors = new Float32Array(
        vecBuf.buffer.slice(vecBuf.byteOffset, vecBuf.byteOffset + vecBuf.byteLength),
      );
      if (meta.version !== 1) throw new Error(`unexpected version ${String(meta.version)}`);
      if (meta.dim <= 0) throw new Error("bad dim");
      if (vectors.length !== chunks.length * meta.dim) {
        throw new Error(
          `vectors.bin has ${vectors.length} floats, expected ${chunks.length} x ${meta.dim}`,
        );
      }
      store.model = meta.model;
      store.provider = meta.provider;
      store.dim = meta.dim;
      store.nextFileId = Math.max(meta.nextFileId, 1);
      store.files = new Map(meta.files.map((f) => [f.path, f]));
      store.chunks = chunks;
      store.vectors = vectors;
      return store;
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      throw new StoreCorruptError(
        `Index in ${dir} is corrupt (${msg}).`,
        "Run clear_index (confirm: true), or set SMART_DOCS_DATA_DIR to a fresh directory.",
      );
    }
  }

  /** Refuse to mix vectors from two different embedding models in one index. */
  assertCompatible(identity: StoreIdentity): void {
    if (this.count === 0) return;
    if (this.model !== identity.model || this.provider !== identity.provider) {
      throw new ModelMismatchError(
        `This index was built with provider "${this.provider}" model "${this.model || "?"}", ` +
          `but the current configuration is provider "${identity.provider}" model "${identity.model}".`,
        "Either restore your SMART_DOCS_EMBEDDING_* environment values, or call clear_index (confirm: true) and index again with the new model.",
      );
    }
  }

  /**
   * Merge one file's chunks + embeddings, replacing any previous version of the
   * same path. On the first insert this also adopts the index identity.
   * `rows` is a flat Float32Array of chunkCount*dim floats.
   */
  add(
    record: Omit<FileRecord, "id" | "chunkCount">,
    chunks: ChunkMeta[],
    rows: Float32Array,
    dim: number,
    identity: StoreIdentity,
  ): void {
    if (chunks.length === 0) return;
    this.assertCompatible(identity);
    if (this.count > 0 && dim !== this.dim) {
      throw new SmartDocsError(
        `Embedding dimension mismatch: index has ${this.dim}, this batch has ${dim}.`,
        "This usually means the embedding model changed mid-run; restart smart-docs.",
      );
    }
    if (rows.length !== chunks.length * dim) {
      throw new SmartDocsError(
        `Embedding row count mismatch for ${record.path}: expected ${chunks.length * dim} floats, got ${rows.length}.`,
      );
    }

    // Drop any previous version of this path first (keeps ids stable otherwise).
    const existing = this.files.get(record.path);
    if (existing) this.removeFile(record.path);

    if (this.count === 0) {
      this.model = identity.model;
      this.provider = identity.provider;
      this.dim = dim;
    }

    const id = this.nextFileId++;
    const rec: FileRecord = { ...record, id, chunkCount: chunks.length };

    const base = this.chunks.length;
    for (let i = 0; i < chunks.length; i++) {
      const meta = { ...chunks[i]!, fileId: id }; // bind chunks to this file's id
      const row = rows.subarray(i * dim, (i + 1) * dim);
      // Defensive re-normalization (cosine expects unit vectors everywhere).
      let norm = 0;
      for (let j = 0; j < row.length; j++) norm += row[j]! * row[j]!;
      norm = Math.sqrt(norm) || 1;
      for (let j = 0; j < row.length; j++) row[j] = row[j]! / norm;
      this.chunks.push(meta);
    }
    const grown = new Float32Array(this.chunks.length * dim);
    grown.set(this.vectors.subarray(0, base * dim), 0);
    grown.set(rows, base * dim);
    this.vectors = grown;

    this.files.set(rec.path, rec);
    this.touch();
  }

  removeFile(path: string): boolean {
    const rec = this.files.get(path);
    if (!rec) return false;
    const kept: ChunkMeta[] = [];
    for (const c of this.chunks) if (c.fileId !== rec.id) kept.push(c);
    const removed = this.chunks.length - kept.length;
    if (removed > 0 && this.dim > 0) {
      const next = new Float32Array(kept.length * this.dim);
      let w = 0;
      for (let i = 0; i < this.chunks.length; i++) {
        if (this.chunks[i]!.fileId === rec.id) continue;
        next.set(this.vectors.subarray(i * this.dim, (i + 1) * this.dim), w);
        w += this.dim;
      }
      this.vectors = next;
    }
    this.chunks = kept;
    this.files.delete(path);
    this.touch();
    return true;
  }

  /**
   * Top-k cosine similarity search with optional file-level filtering.
   * Hits are returned sorted by descending score.
   */
  search(query: Float32Array, k: number, filter?: SearchFilter): SearchHit[] {
    if (this.count === 0 || this.dim === 0) return [];
    if (query.length !== this.dim) {
      throw new SmartDocsError(
        `Query has ${query.length} dimensions but the index has ${this.dim}.`,
        "The embedding model probably changed since indexing; re-run index_directory after clearing.",
      );
    }
    let qn = 0;
    for (let j = 0; j < query.length; j++) qn += query[j]! * query[j]!;
    qn = Math.sqrt(qn);
    if (qn === 0) return [];

    const allowed = filter ? allowedFileIds(this.files, filter) : undefined;
    if (allowed && allowed.size === 0) return [];

    // Simple top-k selection: fine for k<=50 and 10^5 chunks.
    const top: Array<{ idx: number; score: number }> = [];
    let minScore = Infinity;
    let minPos = 0;
    for (let i = 0; i < this.chunks.length; i++) {
      if (allowed && !allowed.has(this.chunks[i]!.fileId)) continue;
      let dot = 0;
      const off = i * this.dim;
      for (let j = 0; j < this.dim; j++) dot += query[j]! * this.vectors[off + j]!;
      const score = dot / qn;
      if (top.length < k) {
        top.push({ idx: i, score });
        if (top.length === k) {
          for (let x = 0; x < top.length; x++) {
            if (top[x]!.score < minScore) {
              minScore = top[x]!.score;
              minPos = x;
            }
          }
        }
      } else if (score > minScore) {
        top[minPos] = { idx: i, score };
        minScore = Infinity;
        for (let x = 0; x < top.length; x++) {
          if (top[x]!.score < minScore) {
            minScore = top[x]!.score;
            minPos = x;
          }
        }
      } else {
        continue;
      }
    }
    top.sort((a, b) => b.score - a.score);
    return top.map(({ idx, score }) => {
      const c = this.chunks[idx]!;
      const rec = this.fileIdMap().get(c.fileId)!;
      return {
        path: rec.path,
        kind: rec.kind,
        score,
        text: c.text,
        title: c.title,
        startLine: c.startLine,
        endLine: c.endLine,
        page: c.page,
      } satisfies SearchHit;
    });
  }

  private fileIdCache: Map<number, FileRecord> | undefined = undefined;

  private fileIdMap(): Map<number, FileRecord> {
    if (!this.fileIdCache || this.fileIdCache.size !== this.files.size) {
      this.fileIdCache = new Map();
      for (const f of this.files.values()) this.fileIdCache.set(f.id, f);
    }
    return this.fileIdCache;
  }

  /** Invalidate cached id map after file mutations. */
  private touch(): void {
    this.fileIdCache = undefined;
  }

  stats(): { model: string; provider: string; dim: number; fileCount: number; chunkCount: number } {
    return {
      model: this.model,
      provider: this.provider,
      dim: this.dim,
      fileCount: this.files.size,
      chunkCount: this.chunks.length,
    };
  }

  /** Drop everything in memory. */
  reset(): void {
    this.files = new Map();
    this.chunks = [];
    this.vectors = new Float32Array(0);
    this.nextFileId = 1;
    this.model = "";
    this.provider = "";
    this.dim = 0;
    this.touch();
  }

  /** Atomically persist meta, chunk metadata and raw vectors to disk. */
  async save(): Promise<void> {
    await fs.mkdir(this.dir, { recursive: true, mode: DIR_MODE });
    const meta: PersistedStoreMeta = {
      version: 1,
      model: this.model,
      provider: this.provider,
      dim: this.dim,
      nextFileId: this.nextFileId,
      files: [...this.files.values()],
    };
    await atomicWriteText(join(this.dir, META_FILE), JSON.stringify(meta, null, 2));
    await atomicWriteText(join(this.dir, CHUNKS_FILE), JSON.stringify(this.chunks));
    const bytes = Buffer.from(this.vectors.buffer, this.vectors.byteOffset, this.vectors.byteLength);
    await atomicWriteBuffer(join(this.dir, VECTORS_FILE), bytes);
  }

  /** Whether a persisted index exists on disk. */
  static async existsOnDisk(dir: string): Promise<boolean> {
    try {
      await fs.access(join(dir, META_FILE));
      return true;
    } catch {
      return false;
    }
  }

  /** Delete persisted files (not the local models cache). */
  static async clearOnDisk(dir: string): Promise<void> {
    await Promise.all(
      [META_FILE, CHUNKS_FILE, VECTORS_FILE].map(async (f) => fs.rm(join(dir, f), { force: true })),
    );
  }
}

function extOf(path: string): string {
  const i = path.lastIndexOf(".");
  const slash = path.lastIndexOf("/");
  return i > slash ? path.slice(i).toLowerCase() : "";
}

function allowedFileIds(files: Map<string, FileRecord>, filter: SearchFilter): Set<number> {
  const prefixes = filter.paths?.length
    ? filter.paths.map((p) => (p.endsWith("/") ? p : p.replace(/\/+$/, "") + "/"))
    : undefined;
  const exts = filter.extensions?.length
    ? new Set(filter.extensions.map((e) => (e.startsWith(".") ? e.toLowerCase() : `.${e.toLowerCase()}`)))
    : undefined;
  const kinds = filter.kinds?.length ? new Set<FileKind>(filter.kinds) : undefined;
  const out = new Set<number>();
  for (const f of files.values()) {
    if (prefixes && !prefixes.some((p) => (f.path.startsWith(p) || f.path + "/" === p))) continue;
    if (exts && !exts.has(extOf(f.path))) continue;
    if (kinds && !kinds.has(f.kind)) continue;
    out.add(f.id);
  }
  return out;
}

/** Index data includes verbatim copies of the user's documents — keep it private. */
const FILE_MODE = 0o600;
const DIR_MODE = 0o700;

async function atomicWriteText(path: string, data: string): Promise<void> {
  const tmp = `${path}.tmp-${process.pid}`;
  await fs.writeFile(tmp, data, { encoding: "utf8", mode: FILE_MODE });
  await fs.rename(tmp, path);
}

async function atomicWriteBuffer(path: string, data: Buffer): Promise<void> {
  const tmp = `${path}.tmp-${process.pid}`;
  await fs.writeFile(tmp, data, { mode: FILE_MODE });
  await fs.rename(tmp, path);
}
