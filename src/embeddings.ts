import { SmartDocsError } from "./errors.js";
import type { Config, RemoteEmbeddingConfig } from "./config.js";

export interface EmbeddingProvider {
  /** Stable identity persisted with the index ("local" | "remote"). */
  readonly provider: string;
  /** Model id persisted with the index. */
  readonly model: string;
  /** Embed a batch of texts; returns one unit-norm Float32Array per text. */
  embed(texts: string[]): Promise<Float32Array[]>;
}

function normalize(v: Float32Array): Float32Array {
  let n = 0;
  for (let i = 0; i < v.length; i++) n += v[i]! * v[i]!;
  n = Math.sqrt(n) || 1;
  for (let i = 0; i < v.length; i++) v[i] = v[i]! / n;
  return v;
}

/**
 * Local, fully-offline embeddings via transformers.js (ONNX, no native build).
 * The model (~25-90 MB, quantized) is downloaded once into
 * `<dataRoot>/models` and afterwards everything runs from disk.
 */
export class LocalEmbeddingProvider implements EmbeddingProvider {
  readonly provider = "local";
  private extractorP: Promise<FeatureExtractor> | undefined;

  constructor(
    readonly model: string,
    private readonly modelCacheDir: string,
    private readonly batchSize = 32,
  ) {}

  private async extractor(): Promise<FeatureExtractor> {
    if (!this.extractorP) {
      this.extractorP = (async () => {
        const { pipeline, env } = await import("@huggingface/transformers");
        env.cacheDir = this.modelCacheDir;
        env.allowLocalModels = false;
        try {
          return (await pipeline("feature-extraction", this.model)) as FeatureExtractor;
        } catch (err) {
          const msg = err instanceof Error ? err.message : String(err);
          throw new SmartDocsError(
            `Could not load local embedding model "${this.model}": ${msg}`,
            "Check the model id (README lists tested options), ensure one-time network access to huggingface.co (the model is cached afterwards), or switch providers: SMART_DOCS_EMBEDDING_PROVIDER=remote.",
          );
        }
      })();
    }
    return this.extractorP;
  }

  async embed(texts: string[]): Promise<Float32Array[]> {
    if (texts.length === 0) return [];
    const extractor = await this.extractor();
    const out: Float32Array[] = [];
    for (let i = 0; i < texts.length; i += this.batchSize) {
      const batch = texts.slice(i, i + this.batchSize);
      const res = await extractor(batch, { pooling: "mean", normalize: true });
      out.push(...rowsOf(res));
    }
    return out;
  }

  async dispose(): Promise<void> {
    if (this.extractorP) {
      const e = await this.extractorP;
      e.dispose?.();
      this.extractorP = undefined;
    }
  }
}

type TensorData = { dims: number[]; data: Uint8Array | Float32Array | Int8Array };

type FeatureExtractor = {
  (texts: string[], opts: { pooling: "mean"; normalize: true }): Promise<TensorData>;
  dispose?: () => void;
};

/** Split a [n, d] tensor into n row Float32Arrays. */
function rowsOf(t: TensorData): Float32Array[] {
  const dims = t.dims;
  if (dims.length !== 2) {
    throw new SmartDocsError(`Unexpected embedding tensor shape ${dims.join("x")} (expected n×d).`);
  }
  const [n, d] = dims as [number, number];
  const rows: Float32Array[] = [];
  for (let i = 0; i < n; i++) {
    const row = new Float32Array(d);
    for (let j = 0; j < d; j++) row[j] = Number(t.data[i * d + j]);
    rows.push(normalize(row));
  }
  return rows;
}

/** Free/cheap OpenAI-compatible /embeddings endpoint (OpenAI, Together, Gemini analogs, local servers like Ollama). */
export class RemoteEmbeddingProvider implements EmbeddingProvider {
  readonly provider = "remote";

  constructor(
    readonly model: string,
    private readonly remote: RemoteEmbeddingConfig,
    private readonly batchSize = 32,
    private readonly requestDelayMs = 250,
  ) {}

  async embed(texts: string[]): Promise<Float32Array[]> {
    if (texts.length === 0) return [];
    const out: Float32Array[] = [];
    for (let i = 0; i < texts.length; i += this.batchSize) {
      const batch = texts.slice(i, i + this.batchSize);
      const rows = await this.embedBatchWithRetry(batch);
      for (const r of rows) out.push(normalize(r));
      if (i + this.batchSize < texts.length) {
        await new Promise((r) => setTimeout(r, this.requestDelayMs)); // rate-limit friendly
      }
    }
    return out;
  }

  private async embedBatchWithRetry(batch: string[], attempts = 3): Promise<Float32Array[]> {
    let lastErr: unknown;
    for (let attempt = 1; attempt <= attempts; attempt++) {
      try {
        return await this.embedBatchOnce(batch);
      } catch (err) {
        lastErr = err;
        if (attempt < attempts) {
          const delay = 1000 * 2 ** (attempt - 1);
          await new Promise((r) => setTimeout(r, delay));
        }
      }
    }
    const msg = lastErr instanceof Error ? lastErr.message : String(lastErr);
    throw new SmartDocsError(
      `Embedding request to ${this.remote.baseUrl} failed after ${attempts} attempts: ${msg}`,
      "Check SMART_DOCS_EMBEDDING_BASE_URL / API key / model name, or revert to local embeddings (SMART_DOCS_EMBEDDING_PROVIDER=local).",
    );
  }

  private async embedBatchOnce(batch: string[]): Promise<Float32Array[]> {
    const headers: Record<string, string> = { "content-type": "application/json" };
    if (this.remote.apiKey) headers.authorization = `Bearer ${this.remote.apiKey}`;
    const res = await fetch(`${this.remote.baseUrl.replace(/\/+$/, "")}/embeddings`, {
      method: "POST",
      headers,
      body: JSON.stringify({ model: this.remote.model, input: batch }),
      signal: AbortSignal.timeout(60_000),
    });
    if (!res.ok) {
      const body = await res.text().catch(() => "");
      throw new Error(`HTTP ${res.status}: ${body.slice(0, 300)}`);
    }
    const json = (await res.json()) as { data?: Array<{ embedding: number[] }> };
    const data = json.data;
    if (!Array.isArray(data) || data.length !== batch.length) {
      throw new Error(`unexpected /embeddings response shape (${res.status})`);
    }
    const rows: Float32Array[] = [];
    for (const item of data) {
      const d = item.embedding;
      if (!Array.isArray(d) || d.length === 0) throw new Error("empty embedding array in response");
      rows.push(Float32Array.from(d));
    }
    return rows;
  }
}

export function createEmbeddingProvider(cfg: Config): EmbeddingProvider {
  if (cfg.embeddingProvider === "remote" && cfg.remoteEmbedding) {
    return new RemoteEmbeddingProvider(cfg.remoteEmbedding.model, cfg.remoteEmbedding);
  }
  return new LocalEmbeddingProvider(cfg.embeddingModel, `${cfg.dataRoot}/models`);
}
