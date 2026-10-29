import type { FileKind } from "./types.js";

/** Names of the doc kinds we know how to index. */
export const FILE_KINDS: readonly FileKind[] = ["markdown", "text", "rst", "pdf", "code"] as const;

/**
 * Default walk exclusions. Directories that are almost never documentation
 * sources; indexing them wastes time and pollutes results.
 */
export const DEFAULT_EXCLUDED_DIRS: readonly string[] = [
  "node_modules",
  ".git",
  ".hg",
  ".svn",
  ".smart-docs",
  ".next",
  ".nuxt",
  ".turbo",
  ".cache",
  "dist",
  "build",
  "out",
  "coverage",
  "target",
  "venv",
  ".venv",
  "__pycache__",
  ".idea",
  ".vscode",
] as const;

/** Everything supported out of the box, grouped by kind. */
export const DEFAULT_EXTENSIONS: readonly string[] = [
  // markdown
  ".md", ".markdown", ".mdx",
  // plain text
  ".txt",
  // rst
  ".rst",
  // pdf
  ".pdf",
  // code (searchable source, comment blocks especially useful)
  ".js", ".jsx", ".ts", ".tsx", ".mts", ".cts", ".cjs", ".mjs",
  ".py", ".rb", ".java", ".cs", ".go", ".rs", ".c", ".h", ".cc", ".cpp", ".hpp",
  ".swift", ".kt", ".kts", ".scala", ".php", ".pl", ".lua", ".dart",
  ".sh", ".bash", ".zsh", ".fish",
  ".sql", ".proto", ".graphql",
  ".toml", ".yaml", ".yml", ".ini", ".cfg", ".properties",
  ".vue", ".svelte", ".astro", ".ejs", ".hbs",
  ".bat", ".ps1",
] as const;

/** Extensionless file basenames we support (Dockerfiles, Makefiles). */
export const SUPPORTED_BASENAMES: readonly string[] = [
  "dockerfile", "makefile", "justfile", "cmakelists.txt",
] as const;

/**
 * Extensions we index as executable config/source metadata only — not prose.
 * (Kept for future special handling; currently treated like code.)
 */
export const CONFIG_EXTENSIONS: readonly string[] = [
  ".editorconfig", ".json", ".jsonc",
] as const;
