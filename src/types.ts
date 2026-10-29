export type FileKind = "markdown" | "text" | "rst" | "pdf" | "code";

/** A chunk produced by the chunker before embedding. */
export interface RawChunk {
  text: string;
  /** 1-based start line in the source text file (0 for PDFs, which use `page`). */
  startLine: number;
  /** 1-based inclusive end line */
  endLine: number;
  /** 1-based page number for PDF sources */
  page?: number;
  /** Markdown/RST heading path this chunk sits under ("" if before any heading). */
  title?: string;
}

/** Anything indexable - file record stored alongside vectors. */
export interface FileRecord {
  /** stable numeric id, never reused */
  id: number;
  /** absolute path */
  path: string;
  kind: FileKind;
  sizeBytes: number;
  mtimeMs: number;
  /** sha256 of raw file bytes */
  contentHash: string;
  chunkCount: number;
  indexedAt: string;
}

/** Metadata for one stored embedding row. */
export interface ChunkMeta {
  fileId: number;
  text: string;
  title?: string;
  startLine: number;
  endLine: number;
  page?: number;
}

export interface PersistedStoreMeta {
  version: 1;
  model: string;
  provider: string;
  dim: number;
  nextFileId: number;
  files: FileRecord[];
}

/** One search hit, already joined with the file record. */
export interface SearchHit {
  path: string;
  kind: FileKind;
  score: number;
  text: string;
  title?: string;
  startLine: number;
  endLine: number;
  page?: number;
}

export interface SearchFilter {
  /** only files whose absolute path starts with one of these prefixes */
  paths?: string[];
  /** only files with one of these extensions (with leading dot, case-insensitive) */
  extensions?: string[];
  kinds?: FileKind[];
}

export interface IndexProgress {
  fileDone: number;
  fileTotal: number;
  currentFile?: string;
}

export type ProgressListener = (p: IndexProgress) => void;
