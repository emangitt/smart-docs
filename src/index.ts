/** Public API surface for programmatic use of smart-docs. */
export { loadConfig, defaultDataRoot, DEFAULT_EMBEDDING_MODEL, type Config } from "./config.js";
export { Services } from "./services.js";
export { createSmartDocsServer, main } from "./server.js";
export { defineTools, type ToolDef } from "./tools.js";
export { VectorStore } from "./store.js";
export { createEmbeddingProvider, LocalEmbeddingProvider, RemoteEmbeddingProvider, type EmbeddingProvider } from "./embeddings.js";
export { chunkExtracted, chunkMarkdown, chunkRst, chunkCode, chunkText, chunkPdfPages, type ChunkOptions } from "./chunker.js";
export { extractFile, extractBytes, fileKindFor, type ExtractedFile } from "./extract.js";
export { VERSION } from "./version.js";
