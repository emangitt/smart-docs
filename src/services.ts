import { join } from "node:path";
import type { Config } from "./config.js";
import { VectorStore } from "./store.js";
import { createEmbeddingProvider, type EmbeddingProvider } from "./embeddings.js";
import { getLlm, type LlmClient } from "./llm.js";
import type { StoreIdentity } from "./store.js";

/**
 * Wires configuration to concrete services. Store is lazy-loaded once and
 * cached; `reloadStore()` re-reads from disk (after clear_index).
 */
export class Services {
  private storeP: Promise<VectorStore> | undefined;

  readonly indexDir: string;

  constructor(readonly cfg: Config) {
    this.indexDir = join(cfg.dataRoot, "index");
  }

  /** Persisted vector store (loaded once per process). */
  store(): Promise<VectorStore> {
    if (!this.storeP) this.storeP = VectorStore.load(this.indexDir);
    return this.storeP;
  }

  /** Discard the cached store and re-load from disk. */
  async reloadStore(): Promise<VectorStore> {
    this.storeP = undefined;
    return this.store();
  }

  async saveStore(): Promise<void> {
    const store = await this.store();
    await store.save();
  }

  /** New embedding provider for the configured settings. */
  createEmbedder(): EmbeddingProvider {
    return createEmbeddingProvider(this.cfg);
  }

  /** Optional abstractive LLM (null => offline extractive mode). */
  llm(): LlmClient | null {
    return getLlm(this.cfg);
  }

  /** The model identity stored in / required by the index. */
  identity(embedder: EmbeddingProvider): StoreIdentity {
    return { model: embedder.model, provider: embedder.provider };
  }
}
