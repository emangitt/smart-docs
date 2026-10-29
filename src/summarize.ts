/** Minimal chunk-like shape needed for extractive summarization. */
export interface PassageLike {
  text: string;
  startLine: number;
  endLine: number;
  page?: number;
  title?: string;
}

export interface SummaryPassage extends PassageLike {
  path: string;
}

const STOPWORDS = new Set(
  ("the a an and or but if then else for to of in on at by with from as is are was were be been being " +
    "this that these those it its not no so such can will just about into over under than they you your our")
    .split(" "),
);

function sentences(text: string): string[] {
  return text
    .replace(/\s+/g, " ")
    .split(/(?<=[.!?])\s+/)
    .map((s) => s.trim())
    .filter((s) => s.length >= 20);
}

function keywords(s: string): string[] {
  return s
    .toLowerCase()
    .split(/[^a-z0-9_]+/)
    .filter((w) => w.length >= 3 && !STOPWORDS.has(w));
}

/**
 * Offline extractive summarization: score sentences by keyword frequency
 * (a light class-based TF heuristic), keep the best ones in document order.
 * No LLM and no network — used directly, and as fallback when no LLM is set.
 */
export function extractiveSummary(
  items: Array<{ meta: PassageLike; path: string }>,
  maxSentences = 8,
): SummaryPassage[] {
  const freq = new Map<string, number>();
  const allSentences: Array<{ s: string; path: string; startLine: number; endLine: number; page?: number; title?: string; pos: number }> = [];
  for (const { meta, path } of items) {
    for (const s of sentences(meta.text)) {
      for (const w of keywords(s).slice(0, 40)) freq.set(w, (freq.get(w) ?? 0) + 1);
      allSentences.push({
        s,
        path,
        startLine: meta.startLine,
        endLine: meta.endLine,
        page: meta.page,
        title: meta.title,
        pos: allSentences.length,
      });
    }
  }
  const scored = allSentences.map((e, idx) => {
    const kws = keywords(e.s);
    let score = 0;
    for (const w of kws) score += freq.get(w) ?? 0;
    score = kws.length > 0 ? score / Math.sqrt(kws.length) : 0;
    score *= idx < 6 ? 1.25 : 1.0; // lead bias
    return { ...e, score };
  });
  return scored
    .sort((a, b) => b.score - a.score)
    .slice(0, maxSentences)
    .sort((a, b) => a.pos - b.pos)
    .map((e) => ({
      text: e.s,
      path: e.path,
      startLine: e.startLine,
      endLine: e.endLine,
      page: e.page,
      title: e.title,
    }));
}

/** Cap joined context length to keep prompts/replies manageable. */
export function clipTotalPassages(passages: SummaryPassage[], maxChars = 12000): SummaryPassage[] {
  const out: SummaryPassage[] = [];
  let used = 0;
  for (const p of passages) {
    if (used + p.text.length > maxChars) break;
    out.push(p);
    used += p.text.length;
  }
  return out;
}
