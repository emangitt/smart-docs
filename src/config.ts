import { homedir } from "node:os";
import { join, resolve } from "node:path";

export class ConfigError extends Error {}

export interface RemoteEmbeddingConfig {
  /** OpenAI-compatible base URL, e.g. https://api.together.xyz/v1 */
  baseUrl: string;
  apiKey?: string;
  model: string;
}

export interface LlmConfig {
  /** OpenAI-compatible base URL, e.g. https://api.openai.com/v1 */
  baseUrl: string;
  apiKey?: string;
  model: string;
}

export interface Config {
  /** Root model + persistence dir; index lives in `<root>/index`. */
  dataRoot: string;
  /** Embedding model id (transformers.js id or provider-specific name). */
  embeddingModel: string;
  /** "local" (default, offline transformers.js) or "remote" (OpenAI-compatible). */
  embeddingProvider: "local" | "remote";
  remoteEmbedding?: RemoteEmbeddingConfig;
  /** Optional LLM for abstractive summaries / ask_docs answers (falls back to extractive). */
  llm?: LlmConfig;
  /** Target chunk size in characters. */
  chunkSize: number;
  /** Overlap between consecutive chunks in characters (approximate). */
  chunkOverlap: number;
  /** Skip files larger than this. */
  maxFileBytes: number;
  /** Safety cap on files indexed per index_directory call. */
  maxFilesPerCall: number;
  /** Extra directory names to skip on top of the built-in list. */
  extraExcludedDirs: string[];
  /** Default k for semantic_search / ask_docs. */
  defaultTopK: number;
}

export const DEFAULT_EMBEDDING_MODEL = "Xenova/all-MiniLM-L6-v2";

export function defaultDataRoot(): string {
  const custom = process.env.SMART_DOCS_DATA_DIR;
  if (custom && custom.trim()) return resolve(custom.trim());
  return join(homedir(), ".smart-docs");
}

function num(v: string | undefined): number | undefined {
  if (v === undefined || v.trim() === "") return undefined;
  const n = Number(v);
  return Number.isFinite(n) ? n : undefined;
}

function parseArgs(argv: string[]): Map<string, string | true> {
  const map = new Map<string, string | true>();
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i]!;
    if (!a.startsWith("--")) continue;
    const eq = a.indexOf("=");
    if (eq > 0) {
      map.set(a.slice(2, eq), a.slice(eq + 1));
    } else if (argv[i + 1] !== undefined && !argv[i + 1]!.startsWith("--")) {
      map.set(a.slice(2), argv[i + 1]!);
      i++;
    } else {
      map.set(a.slice(2), true);
    }
  }
  return map;
}

/** Parse tool/environment/CLI configuration. Throws ConfigError with useful messages. */
export function loadConfig(argv: string[] = [], env: NodeJS.ProcessEnv = process.env): Config {
  const flags = parseArgs(argv);

  if (flags.has("help")) {
    throw new ConfigError(`smart-docs — MCP server for local doc search. Starts on stdio.\n
Options:
  --index-dir <dir>      storage root (default ~/.smart-docs)
  --model <id>           embedding model (default: ${DEFAULT_EMBEDDING_MODEL})
  --chunk-size <chars>   target chunk size (default 1200)
  --overlap <chars>      chunk overlap (default 200)
  --help                 show this help
See README for environment variables and MCP client config.`);
  }

  const dataRoot = (() => {
    if (typeof flags.get("index-dir") === "string") return resolve(flags.get("index-dir") as string);
    if (env.SMART_DOCS_DATA_DIR?.trim()) return resolve(env.SMART_DOCS_DATA_DIR.trim());
    return join(homedir(), ".smart-docs");
  })();

  const model =
    (typeof flags.get("model") === "string" ? (flags.get("model") as string) : undefined) ??
    env.SMART_DOCS_EMBEDDING_MODEL?.trim() ??
    DEFAULT_EMBEDDING_MODEL;

  const provider = (env.SMART_DOCS_EMBEDDING_PROVIDER?.trim().toLowerCase() === "remote"
    ? "remote"
    : "local") as "local" | "remote";

  let remoteEmbedding: RemoteEmbeddingConfig | undefined;
  if (provider === "remote") {
    const baseUrl = env.SMART_DOCS_EMBEDDING_BASE_URL?.trim();
    if (!baseUrl) {
      throw new ConfigError(
        "SMART_DOCS_EMBEDDING_PROVIDER=remote requires SMART_DOCS_EMBEDDING_BASE_URL (OpenAI-compatible base URL, e.g. https://api.openai.com/v1).",
      );
    }
    remoteEmbedding = {
      baseUrl,
      apiKey: env.SMART_DOCS_EMBEDDING_API_KEY?.trim() || undefined,
      model: env.SMART_DOCS_EMBEDDING_REMOTE_MODEL?.trim() || "",
    };
    if (!remoteEmbedding.model) {
      throw new ConfigError(
        "remote embeddings require SMART_DOCS_EMBEDDING_REMOTE_MODEL (e.g. text-embedding-3-small).",
      );
    }
  }

  let llm: LlmConfig | undefined;
  const llmBase = env.SMART_DOCS_LLM_BASE_URL?.trim();
  if (llmBase) {
    llm = {
      baseUrl: llmBase,
      apiKey: env.SMART_DOCS_LLM_API_KEY?.trim() || undefined,
      model: env.SMART_DOCS_LLM_MODEL?.trim() || "",
    };
    if (!llm.model) {
      throw new ConfigError("SMART_DOCS_LLM_BASE_URL requires SMART_DOCS_LLM_MODEL (e.g. gpt-4o-mini).");
    }
  }

  const chunkSize = num(env.SMART_DOCS_CHUNK_SIZE) ?? 1200;
  const chunkOverlap = num(env.SMART_DOCS_CHUNK_OVERLAP) ?? 200;
  const maxFileBytes = (num(env.SMART_DOCS_MAX_FILE_MB) ?? 10) * 1024 * 1024;
  const maxFilesPerCall = num(env.SMART_DOCS_MAX_FILES) ?? 2000;
  const defaultTopK = num(env.SMART_DOCS_TOP_K) ?? 8;
  const extraExcludedDirs = (env.SMART_DOCS_EXCLUDE_DIRS ?? "")
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);

  if (chunkSize < 100) throw new ConfigError("SMART_DOCS_CHUNK_SIZE must be at least 100 characters.");
  if (chunkOverlap < 0 || chunkOverlap >= chunkSize) {
    throw new ConfigError("SMART_DOCS_CHUNK_OVERLAP must be >= 0 and smaller than chunk size.");
  }
  if (maxFileBytes < 1024) throw new ConfigError("SMART_DOCS_MAX_FILE_MB must be at least 0.001 MB.");
  if (maxFilesPerCall < 1) throw new ConfigError("SMART_DOCS_MAX_FILES must be at least 1.");
  if (defaultTopK < 1 || defaultTopK > 50) throw new ConfigError("SMART_DOCS_TOP_K must be 1..50.");

  return {
    dataRoot,
    embeddingModel: model,
    embeddingProvider: provider,
    remoteEmbedding,
    llm,
    chunkSize,
    chunkOverlap,
    maxFileBytes,
    maxFilesPerCall,
    extraExcludedDirs,
    defaultTopK,
  };
}
