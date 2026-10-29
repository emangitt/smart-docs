import { promises as fs } from "node:fs";
import { extname } from "node:path";
import {
  BinaryFileError,
  EmptyFileError,
  FileTooLargeError,
  SmartDocsError,
} from "./errors.js";
import { DEFAULT_EXTENSIONS, SUPPORTED_BASENAMES } from "./constants.js";
import type { FileKind } from "./types.js";

export interface ExtractedFile {
  kind: FileKind;
  /** Full text: LF-normalized, BOM stripped, frontmatter separated (markdown). */
  text: string;
  /** YAML frontmatter (markdown only, without the --- fences). */
  frontmatter?: string;
  /** Per-page text for PDF sources. */
  pages?: Array<{ page: number; text: string }>;
}

/** Map a path to its doc kind; undefined when the type is not supported. */
export function fileKindFor(path: string): FileKind | undefined {
  const base = path.split("/").pop()!.toLowerCase();
  if (SUPPORTED_BASENAMES.includes(base)) {
    return base === "cmakelists.txt" ? "text" : "code";
  }
  const ext = extname(path).toLowerCase();
  if ([".md", ".markdown", ".mdx"].includes(ext)) return "markdown";
  if (ext === ".rst") return "rst";
  if (ext === ".pdf") return "pdf";
  if (ext === ".txt") return "text";
  if (DEFAULT_EXTENSIONS.includes(ext)) return "code";
  return undefined;
}

/** The extensions covered out of the box (lowercase, with dot). */
export function defaultExtensions(): string[] {
  return DEFAULT_EXTENSIONS.map((e) => e.toLowerCase());
}

/** Extensionless basenames covered out of the box. */
export function supportedBasenames(): string[] {
  return [...SUPPORTED_BASENAMES];
}

/** Extract text from an already-read file buffer. */
export async function extractBytes(
  path: string,
  buf: Buffer,
  kind: FileKind,
  maxFileBytes: number,
): Promise<ExtractedFile> {
  if (buf.length > maxFileBytes) {
    throw new FileTooLargeError(
      `${path} is ${(buf.length / 1024 / 1024).toFixed(1)} MB, over the ${(maxFileBytes / 1024 / 1024).toFixed(0)} MB limit.`,
      "Raise SMART_DOCS_MAX_FILE_MB or split the file; huge PDFs take a long time to embed.",
    );
  }
  if (buf.length === 0) throw new EmptyFileError(`${path} is empty.`);

  if (kind === "pdf") {
    if (buf.subarray(0, 5).toString("utf8") !== "%PDF-") {
      throw new BinaryFileError(
        `${path} claims .pdf but has no %PDF- header.`,
        "Re-export the file as a real PDF, or exclude it from the index.",
      );
    }
    return extractPdf(buf);
  }

  // Binary sniff: a NUL byte in the first 8 KiB strongly implies binary content.
  if (buf.subarray(0, 8192).includes(0)) {
    throw new BinaryFileError(
      `${path} looks binary (NUL byte found).`,
      "Convert it to a text format (e.g. export .md) or exclude it; smart-docs indexes text sources.",
    );
  }

  const text = buf.toString("utf8").replace(/^\uFEFF/, "").replace(/\r\n?/g, "\n");
  if (!text.trim()) throw new EmptyFileError(`${path} has no text content.`);

  if (kind === "markdown") {
    const fm = stripFrontmatter(text);
    return { kind, text: fm.text, frontmatter: fm.frontmatter };
  }
  return { kind, text };
}

/** Convenience wrapper: stat + read + extract in one go. */
export async function extractFile(path: string, maxFileBytes: number): Promise<ExtractedFile> {
  const kind = fileKindFor(path);
  if (!kind) {
    throw new SmartDocsError(
      `Unsupported file type: ${path}.`,
      "Add the extension via index_directory's `include` option, or implement an extractor (README § Extending file types).",
    );
  }
  const buf = await fs.readFile(path).catch(() => {
    throw new SmartDocsError(`Could not read ${path}.`, "The file may have been moved or deleted; re-run indexing.");
  });
  return extractBytes(path, buf, kind, maxFileBytes);
}

function stripFrontmatter(text: string): { text: string; frontmatter?: string } {
  const m = /^---[ \t]*\n([\s\S]*?)\n---[ \t]*\n?/.exec(text);
  if (!m) return { text };
  const frontmatter = (m[1] ?? "").trim();
  return { text: text.slice(m[0].length), frontmatter: frontmatter || undefined };
}

/** PDFs via unpdf (pure-JS PDF.js wrapper; no native dependencies). */
async function extractPdf(buf: Buffer): Promise<ExtractedFile> {
  const { getDocumentProxy, extractText } = await import("unpdf");
  const doc = await getDocumentProxy(new Uint8Array(buf));
  const { text } = await extractText(doc, { mergePages: false });
  const pages = text.map((t, i) => ({ page: i + 1, text: t.replace(/\r\n?/g, "\n") }));
  const joined = pages.map((p) => p.text).join("\n\n");
  if (!joined.trim()) {
    throw new SmartDocsError(
      "PDF has no extractable text.",
      "It may be scanned images; smart-docs does not OCR. Export the content as text/markdown instead.",
    );
  }
  return { kind: "pdf", text: joined, pages };
}
