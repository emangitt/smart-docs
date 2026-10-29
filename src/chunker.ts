import type { RawChunk } from "./types.js";

export interface ChunkOptions {
  /** target chunk size in characters */
  chunkSize: number;
  /** approximate overlap between consecutive chunks, in characters */
  chunkOverlap: number;
  /** safety cap so one pathological file can't blow up the index */
  maxChunksPerFile?: number;
}

const DEFAULT_MAX_CHUNKS = 5000;

/** Internal unit: a paragraph/section fragment with its line span. */
interface Unit {
  text: string;
  /** 1-based start line in the file (page-relative for PDFs) */
  start: number;
  /** 1-based inclusive end line */
  end: number;
  /** Section title (markdown/RST heading path). */
  title?: string;
  page?: number;
}

function unit(text: string, start: number, end: number, title?: string, page?: number): Unit {
  return { text, start, end, title, page };
}

function blank(line: string): boolean {
  return line.trim().length === 0;
}

/** Split `lines` into blank-line separated paragraph units. */
function paragraphUnits(lines: string[]): Unit[] {
  const units: Unit[] = [];
  let start = -1;
  let buf: string[] = [];
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]!;
    if (blank(line)) {
      if (start >= 0) {
        const text = buf.join("\n");
        if (text.trim()) units.push(unit(text, start, start + buf.length - 1));
      }
      start = -1;
      buf = [];
      continue;
    }
    if (start < 0) {
      start = i + 1;
      buf = [];
    }
    buf.push(line);
  }
  if (start >= 0) {
    const text = buf.join("\n");
    if (text.trim()) units.push(unit(text, start, start + buf.length - 1));
  }
  return units;
}

/**
 * Merge units into chunks with a sliding window:
 *  - accumulate units until ~chunkSize chars, then flush
 *  - carry trailing units (≈chunkOverlap chars) into the next chunk so line
 *    numbers stay accurate and context survives chunk borders
 *  - single units larger than chunkSize are split on word boundaries with
 *    char overlap; resulting pieces share the unit's line span.
 */
function windowUnits(units: Unit[], opts: ChunkOptions): RawChunk[] {
  const { chunkSize, chunkOverlap } = opts;
  const cap = opts.maxChunksPerFile ?? DEFAULT_MAX_CHUNKS;
  const chunks: RawChunk[] = [];
  let cur: Unit[] = [];
  let curLen = 0;
  let stopped = false;

  const flush = () => {
    if (cur.length === 0 || stopped) return;
    const text = cur.map((u) => u.text.trim()).filter(Boolean).join("\n");
    if (!text.trim()) return;
    const first = cur[0]!;
    // nearest preceding heading (first titled unit) wins as the chunk title
    const titled = cur.find((u) => u.title !== undefined);
    chunks.push({
      text,
      startLine: first.start,
      endLine: cur[cur.length - 1]!.end,
      title: titled?.title,
      page: first.page,
    });
    if (chunks.length >= cap) stopped = true;
  };

  const carryTail = (): Unit[] => {
    if (chunkOverlap <= 0) return []; // no duplication requested
    const carry: Unit[] = [];
    let len = 0;
    for (let i = cur.length - 1; i >= 0; i--) {
      const u = cur[i]!;
      // always carry at least one unit so context survives big gaps
      if (carry.length > 0 && len + u.text.length > chunkOverlap) break;
      carry.unshift(u);
      len += u.text.length;
    }
    return carry;
  };

  for (const original of units) {
    if (stopped) break;
    const pieces = splitOversized(original, chunkSize, chunkOverlap);
    for (const piece of pieces) {
      if (stopped) break;
      if (curLen > 0 && curLen + piece.text.length > chunkSize) {
        const carry = carryTail();
        flush();
        if (stopped) break;
        cur = carry;
        curLen = carry.reduce((n, u) => n + u.text.length, 0);
      }
      cur.push(piece);
      curLen += piece.text.length;
    }
  }
  flush();
  return chunks;
}

/** Word-boundary splitting for units larger than chunkSize. */
function splitOversized(u: Unit, chunkSize: number, overlap: number): Unit[] {
  if (u.text.length <= chunkSize * 1.2) return [u];
  const words = u.text.split(/\s+/).filter(Boolean);
  const pieces: Unit[] = [];
  let buf = "";
  for (const w of words) {
    if (buf.length + w.length > chunkSize && buf.trim()) {
      pieces.push(unit(buf.trim(), u.start, u.end, u.title, u.page));
      const tail = overlap > 0 ? buf.slice(-overlap) : "";
      buf = tail;
    }
    buf += (buf ? " " : "") + w;
  }
  if (buf.trim()) pieces.push(unit(buf.trim(), u.start, u.end, u.title, u.page));
  return pieces.length > 0 ? pieces : [u];
}

/** Markdown: heading-stack titles; headings start new units; frontmatter keeps a title. */
export function chunkMarkdown(text: string, opts: ChunkOptions, frontmatter?: string): RawChunk[] {
  const lines = text.split("\n");
  const units: Unit[] = [];
  const stack: Array<{ level: number; text: string }> = [];

  if (frontmatter?.trim()) {
    for (const p of paragraphUnits(frontmatter.split("\n"))) {
      units.push({ ...p, title: frontmatter.trim().slice(0, 80) });
    }
  }

  let paraStart = -1;
  let paraBuf: string[] = [];
  const flushPara = () => {
    if (paraStart < 0) return;
    const joined = paraBuf.join("\n");
    const title = stack.length > 0 ? stack[stack.length - 1]!.text : undefined;
    if (joined.trim()) units.push(unit(joined, paraStart, paraStart + paraBuf.length - 1, title));
    paraBuf = [];
    paraStart = -1;
  };

  const headingRe = /^(#{1,6})\s+(.+?)\s*#*\s*$/;
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]!;
    const m = headingRe.exec(line);
    if (m) {
      flushPara();
      const level = m[1]!.length;
      while (stack.length > 0 && stack[stack.length - 1]!.level >= level) stack.pop();
      stack.push({ level, text: m[2]!.trim() });
      units.push(unit(line, i + 1, i + 1, m[2]!.trim()));
      continue;
    }
    if (blank(line)) {
      flushPara();
      continue;
    }
    if (paraStart < 0) paraStart = i + 1;
    paraBuf.push(line);
  }
  flushPara();

  return windowUnits(units, opts);
}

/** Look up a heading title if `line` is underlined (reStructuredText style). */
function rstHeadingAt(i: number, lines: string[]): { title: string; level: number } | undefined {
  const underline = /^([=\-`:'"~^_*+#.])\1{2,}$/;
  const prev = i > 0 ? lines[i - 1]! : "";
  const next = lines[i + 1] ?? "";
  const line = lines[i]!;
  if (blank(line) || blank(prev) && blank(next)) return undefined;
  const rank = "=~-^\"'*+#_.:`";
  const levelOf = (c: string): number => {
    const idx = rank.indexOf(c);
    return idx < 0 ? 99 : idx;
  };
  // underline directly below
  if (next.trim() && underline.test(next.trim()) && line.trim() && line.trim().length <= next.trim().length && !underline.test(line.trim())) {
    return { title: line.trim(), level: levelOf(next.trim()[0]!) };
  }
  // underline above (overlined title)
  if (i >= 2 && prev.trim() && underline.test(prev.trim()) && !underline.test(line.trim())) {
    const before = lines[i - 2] ?? "";
    if (blank(before)) return { title: line.trim(), level: levelOf(prev.trim()[0]!) };
  }
  return undefined;
}

/** reStructuredText: headings via underlines, paragraphs otherwise. */
export function chunkRst(text: string, opts: ChunkOptions): RawChunk[] {
  const lines = text.split("\n");
  const units: Unit[] = [];
  const stack: Array<{ level: number; text: string }> = [];
  let i = 0;
  while (i < lines.length) {
    const h = rstHeadingAt(i, lines);
    if (h) {
      while (stack.length > 0 && stack[stack.length - 1]!.level >= h.level) stack.pop();
      stack.push({ level: h.level, text: h.title });
      units.push(unit(lines[i]!, i + 1, i + 1, h.title));
      i += 1;
      continue;
    }
    // collect a paragraph slice
    let j = i;
    while (j < lines.length && !blank(lines[j]!) && !rstHeadingAt(j, lines)) j += 1;
    const slice = lines.slice(i, j);
    const joined = slice.join("\n");
    const title = stack.length > 0 ? stack[stack.length - 1]!.text : undefined;
    if (joined.trim()) units.push(unit(joined, i + 1, i + slice.length, title));
    i = j;
    // skip blank line
    while (i < lines.length && blank(lines[i]!)) i += 1;
  }
  return windowUnits(units, opts);
}

/** Plain text: paragraphs only, no titles. */
export function chunkText(text: string, opts: ChunkOptions): RawChunk[] {
  return windowUnits(paragraphUnits(text.split("\n")), opts);
}

/** Code: blank-line separated blocks; doc comments kept attached for search. */
export function chunkCode(text: string, opts: ChunkOptions): RawChunk[] {
  const lines = text.split("\n");
  const units: Unit[] = [];
  let commentStart = -1;
  let commentBuf: string[] = [];

  const flushComment = (throughLine: number) => {
    const start = commentStart;
    if (start < 0) return;
    const comment = commentBuf.join("\n").trim();
    commentBuf = [];
    commentStart = -1;
    if (comment) {
      units.push(unit(comment, start, throughLine, comment.split("\n")[0]!.slice(0, 100)));
    }
  };

  const isCommentLine = (line: string): boolean => {
    const t = line.trim();
    if (!t) return false;
    if (t.startsWith("#!")) return false; // shebang
    return (
      t.startsWith("//") ||
      t.startsWith("/*") ||
      t.startsWith("*") ||
      t.startsWith("--") ||
      t.startsWith('"""') ||
      t.startsWith("'''") ||
      t.startsWith("#")
    );
  };

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]!;
    if (blank(line)) {
      flushComment(i); // comment ends at the blank line
      continue;
    }
    if (isCommentLine(line)) {
      if (commentStart < 0) commentStart = i + 1;
      commentBuf.push(line);
      continue;
    }
    flushComment(i);
    units.push(unit(line, i + 1, i + 1));
  }
  flushComment(lines.length);
  return windowUnits(units, opts);
}

/** PDF: per-page paragraphs; pages carry `page` numbers instead of file lines. */
export function chunkPdfPages(
  pages: Array<{ page: number; text: string }>,
  opts: ChunkOptions,
): RawChunk[] {
  const maxChunks = opts.maxChunksPerFile ?? DEFAULT_MAX_CHUNKS;
  const perPage = Math.max(1, Math.ceil(maxChunks / Math.max(1, pages.length)));
  const out: RawChunk[] = [];
  for (const p of pages) {
    const pageChunks = windowUnits(paragraphUnits(p.text.split("\n")), { ...opts, maxChunksPerFile: perPage });
    for (const c of pageChunks) out.push({ ...c, page: p.page });
  }
  return out;
}

/** Dispatch on doc kind. */
export function chunkExtracted(
  kind: "markdown" | "text" | "rst" | "pdf" | "code",
  data: { text: string; frontmatter?: string; pages?: Array<{ page: number; text: string }> },
  opts: ChunkOptions,
): RawChunk[] {
  const merged: ChunkOptions = { maxChunksPerFile: DEFAULT_MAX_CHUNKS, ...opts };
  switch (kind) {
    case "markdown":
      return chunkMarkdown(data.text, merged, data.frontmatter);
    case "rst":
      return chunkRst(data.text, merged);
    case "pdf": {
      if (!data.pages) throw new Error("PDF chunks require extracted pages");
      return chunkPdfPages(data.pages, merged);
    }
    case "code":
      return chunkCode(data.text, merged);
    case "text":
    default:
      return chunkText(data.text, merged);
  }
}
