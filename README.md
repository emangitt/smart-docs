# smart-docs

**Local-first semantic search + Q&A over your documentation — for any MCP-compatible AI client.**

smart-docs is a [Model Context Protocol](https://modelcontextprotocol.io/) server that indexes your local docs (Markdown, reStructuredText, plain text, PDFs and common code files) into an **offline vector index**, so agents can find, cite, and answer from the *real* documentation on your machine — not paraphrased training-data guesses.

All data stays on your machine. After the (one-time) embedding-model download, **everything runs fully offline**.

```
index_directory  →  semantic_search  →  ask_docs  →  get_file_summary  →  list_indexed  →  clear_index
```

## Highlights

- **6 focused tools** with strict Zod schemas and recovery-oriented error messages
- **Works offline**: local ONNX embeddings via transformers.js (no GPU, no native builds), vectors stored on disk
- **Smart chunking**: markdown/RST heading-aware sections with configurable size + overlap; code doc-comments kept intact; PDFs split per page
- **Incremental**: unchanged files are skipped by content hash — safe to re-run after every edit
- **Citations everywhere**: results include `path`, section title, `lines` or `page`, similarity score
- **Progress reporting** for long indexing jobs (MCP progress notifications + log messages)
- **Rate-limit friendly / memory-bound**: batched embedding with per-run cache, checkpointed saves, hard file-count/file-size caps
- **Zero native dependencies** — pure TypeScript (strict mode) + pure-JS PDF extraction; no compile step for users

---

## Install

**One command (recommended):**

```bash
npm install -g smart-docs
```

Then point your MCP client at it (below). Or just use `npx` in the config — most clients will fetch it automatically:

```json
{ "command": "npx", "args": ["-y", "smart-docs"] }
```

Requirements: Node.js ≥ 20.11. The first `index_directory` call downloads a small embedding model (~25–90 MB, quantized) once into `~/.smart-docs/models` and verifies nothing afterwards.

---

## Client configuration

### Claude Desktop

`Settings → Developer → Edit Config` (macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`; Windows: `%APPDATA%\Claude\claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "smart-docs": {
      "command": "npx",
      "args": ["-y", "smart-docs"],
      "env": {
        "SMART_DOCS_DATA_DIR": "/Users/yourname/.smart-docs",
        "SMART_DOCS_CHUNK_SIZE": "1200",
        "SMART_DOCS_CHUNK_OVERLAP": "200"
      }
    }
  }
}
```

Or with a pinned global install:

```json
{
  "mcpServers": {
    "smart-docs": {
      "command": "smart-docs",
      "args": []
    }
  }
}
```

### Cursor

Project-level `.cursor/mcp.json` (same shape):

```json
{
  "mcpServers": {
    "smart-docs": {
      "command": "npx",
      "args": ["-y", "smart-docs"],
      "env": {}
    }
  }
}
```

### FreeBuff / any other MCP client

smart-docs is a plain stdio MCP server, so the generic `mcpServers` JSON above works in any client that supports MCP (FreeBuff, VS Code, Raycast, LM Studio, …). Restart the client after editing config.

### Verify it works

Ask the agent:

> Use list_indexed to show what smart-docs has indexed.

Then:

> Use index_directory on ~/code/my-project/docs, then ask_docs: “how is API authentication configured?”

---

## Tools

| Tool | What it does |
|---|---|
| `index_directory` | Recursively scan a folder, extract text (`.md `.mdx `.txt` `.rst` `.pdf` + common code files), chunk, embed, store. Incremental. Progress notifications. If the per-call file cap truncates discovery, the result sets `filesTruncated: true` with a `truncationHint` (raise `SMART_DOCS_MAX_FILES` or index subfolders in multiple calls). |
| `semantic_search` | Natural-language search; returns chunks with `path`, section/lines/page, score, text. Optional `paths` / `extensions` / `kinds` filters. |
| `ask_docs` | Answer a question using **only** indexed knowledge, with `[n]` citations back to source files. |
| `get_file_summary` | Summarize one file (or a whole directory); unindexed files are auto-indexed first. |
| `list_indexed` | Model/provider/dimension, per-kind counts, full file list with chunk counts. |
| `clear_index` | Reset (requires `confirm: true` — a deliberate safety guard). Model cache is kept. |

### Offline vs. LLM mode

- **Default (no LLM configured):** `ask_docs` and `get_file_summary` return *extractive* answers — actual sentences pulled from your files with citations. 100% offline.
- **With an LLM configured** (any OpenAI-compatible endpoint): the same retrieval results are fed as context and you get a *synthesized* answer with citations. Configure via `SMART_DOCS_LLM_*` env vars below. Everything else stays offline.

## Configuration (env vars / CLI flags)

| Env var | CLI flag | Default | Purpose |
|---|---|---|---|
| — | `--help` | | Show usage |
| `SMART_DOCS_DATA_DIR` | `--index-dir <dir>` | `~/.smart-docs` | Where the index (`/index/`) and model cache (`/models/`) live |
| `SMART_DOCS_EMBEDDING_MODEL` | `--model <id>` | `Xenova/all-MiniLM-L6-v2` | Embedding model (transformers.js id) |
| `SMART_DOCS_EMBEDDING_PROVIDER` | | `local` | `local` or `remote` |
| `SMART_DOCS_EMBEDDING_BASE_URL` | | – | OpenAI-compatible base URL (only for `remote`) |
| `SMART_DOCS_EMBEDDING_API_KEY` | | – | Bearer key for remote embeddings |
| `SMART_DOCS_EMBEDDING_REMOTE_MODEL` | | – | Remote embedding model, e.g. `text-embedding-3-small` |
| `SMART_DOCS_CHUNK_SIZE` | `--chunk-size` | `1200` | Target chunk size (characters) |
| `SMART_DOCS_CHUNK_OVERLAP` | `--overlap` | `200` | Chunk overlap (characters, < chunk size) |
| `SMART_DOCS_MAX_FILE_MB` | | `10` | Skip files larger than this |
| `SMART_DOCS_MAX_FILES` | | `2000` | Cap files per `index_directory` call |
| `SMART_DOCS_TOP_K` | | `8` | Default number of search results |
| `SMART_DOCS_EXCLUDE_DIRS` | | – | Extra dir names to skip (comma-separated) |
| `SMART_DOCS_LLM_BASE_URL` | | – | Optional LLM for abstractive answers (OpenAI-compatible) |
| `SMART_DOCS_LLM_API_KEY` | | – | Bearer key for the LLM |
| `SMART_DOCS_LLM_MODEL` | | – | e.g. `gpt-4o-mini`, `llama3.1` (Ollama) |

Important: the embedding model is bound to the index. If you change `SMART_DOCS_EMBEDDING_*`, `index_directory` and `semantic_search` refuse to run with a clear message until you either restore the setting or call `clear_index` — vectors from two models are never mixed.

### Choosing a different local model

Tested local models (all cached offline after first download):

| Model | Dim | Notes |
|---|---|---|
| `Xenova/all-MiniLM-L6-v2` *(default)* | 384 | Fast, great general English default |
| `Xenova/bge-small-en-v1.5` | 384 | Strong retrieval quality |
| `Xenova/jina-embeddings-v2-small-en` | 512 | Good long-context |
| `Xenova/multilingual-e5-small` | 384 | Multilingual docs |

Set it once: `"SMART_DOCS_EMBEDDING_MODEL": "Xenova/bge-small-en-v1.5"` → `clear_index` → re-`index_directory`.

### Free-but-API remote embeddings

Any OpenAI-compatible `/v1/embeddings` endpoint works (local Ollama/LM Studio, Together, OpenAI, …):

```json
"env": {
  "SMART_DOCS_EMBEDDING_PROVIDER": "remote",
  "SMART_DOCS_EMBEDDING_BASE_URL": "https://api.together.xyz/v1",
  "SMART_DOCS_EMBEDDING_API_KEY": "…",        // keep out of chat/repo
  "SMART_DOCS_EMBEDDING_REMOTE_MODEL": "togethercomputer/m2-bert-80M-8k-retrieval"
}
```

Remote mode includes automatic retries with exponential backoff and pacing between batches — index one folder before deciding on speed.

## Usage examples (from your agent)

```
> index_directory { "path": "/Users/me/code/api/docs" }
→ Index root … Added 42 files, 358 chunks (123 skipped unchanged) …

> semantic_search { "query": "how are webhook signatures verified?" }
→ results: [{ "path": ".../webhooks.md", "section": "Signature verification",
              "lines": "28-44", "score": 0.61, "text": "…" }]

> ask_docs { "question": "what rate limits apply to webhooks?" }
→ answer (cited [1], [2]) + sources list

> get_file_summary { "path": "/Users/me/code/api/docs/webhooks.md" }

> get_file_summary { "path": "/Users/me/code/api/docs" }   // directory → lead chunks of every file

> list_indexed
> clear_index { "confirm": true }     // refuses without confirm
```

## Extending file types

Every supported format is one entry in a dispatch table on the kind returned by a file-type probe:

1. **Just include an extension** — unknown-but-textual extensions can be passed directly:

   ```json
   "index_directory": { "path": "~/notes", "include": [".adoc", ".tex", ".carbide"] }
   ```
   These are chunked as plain text (paragraph + sliding window with line numbers).

2. **Add a real extractor with structure awareness** (that's what `.md` heading-awareness gets you). In `src/extract.ts`, map the extension to a `FileKind`; in `src/chunker.ts`, implement a `chunkX(text, opts)` function that yields `RawChunk[]` (`text`, `startLine`, `endLine`, optional `title`/`page`) and register it in `chunkExtracted`'s switch. Suggested next formats: `.adoc` (_asciidoc headings_), `.tex` (sections), `.html` (strip tags, headings), `.docx` (via a docx-to-text library), music-notation? Please contribute.

`fileKindFor()` in `src/extract.ts` is the single source of truth for extension→kind mapping; `DEFAULT_EXTENSIONS`/`SUPPORTED_BASENAMES` in `src/constants.ts` list everything covered out of the box.

## Development

```bash
npm install     # dev deps + runtime deps
npm run build   # typecheck + emit dist/ (strict TypeScript)
npm test        # vitest: chunker, vector store, all 6 tools (stub embeddings, fully offline)
npm run dev     # watch mode
```

Run a server manually against the MCP Inspector:

```bash
npx @modelcontextprotocol/inspector node dist/cli.js
```

Project layout:

```
src/
  cli.ts        # bin entry (stdio)
  server.ts     # MCP wiring + instructions
  tools.ts      # the 6 tools incl. schemas, filters, progress, error hints
  services.ts   # config → store/embedder/LLM wiring
  config.ts     # env/CLI parsing + validation
  store.ts      # flat cosine vector index, atomic persistence, filters
  chunker.ts    # markdown / RST / text / code / PDF chunking
  extract.ts    # file reading, binary sniffing, frontmatter, PDF text
  embeddings.ts # local transformers.js + OpenAI-compatible remote, retries
  summarize.ts  # offline extractive summarizer w/ citations
  llm.ts        # optional OpenAI-compatible client for ask_docs/summaries
test/           # vitest suites (25 tests, no network needed)
```

## What's stored where

- `~/.smart-docs/index/` — `meta.json` (files, model identity), `chunks.json` (chunk text + locations), `vectors.bin` (float32 embeddings). Write-ahead via temp+rename.
- `~/.smart-docs/models/` — cached embedding weights. Deleting this only costs a re-download next run.

## License & the future (Pro)

MIT — see [LICENSE](LICENSE). Free forever, no telemetry, no accounts.

A commercial **smart-docs Pro** is planned and intentionally not in this codebase; it would add:

- **Cloud sync + team indexes** (shared, permissioned doc indexes for whole teams)
- **Higher-quality embeddings** (largest models, hybrid BM25 + vector search, reranking) with a hosted evaluation harness
- **Git-aware indexing** (auto re-index on branch switch, blame-aware citations, PR context)
- **OCR + scanned-PDF premium pipeline**, Office/LaTeX/Confluence extractors
- **Watch mode**: local daemon re-indexes on save without agent calls
- **Project management UI** and enterprise SSO

If those sound useful, the free edition is the right place to start today: the tool contract and index format are designed to migrate.

---

Built with the official [`@modelcontextprotocol/sdk`](https://www.npmjs.com/package/@modelcontextprotocol/sdk), [`@huggingface/transformers`](https://www.npmjs.com/package/@huggingface/transformers) (ONNX runtime), [`unpdf`](https://www.npmjs.com/package/unpdf), and [`zod`](https://www.npmjs.com/package/zod).
