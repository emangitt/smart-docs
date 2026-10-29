import { describe, expect, it } from "vitest";
import {
  chunkCode,
  chunkExtracted,
  chunkMarkdown,
  chunkRst,
  chunkText,
  type ChunkOptions,
} from "../src/chunker.js";

const opts: ChunkOptions = { chunkSize: 300, chunkOverlap: 50, maxChunksPerFile: 100 };
const small: ChunkOptions = { chunkSize: 80, chunkOverlap: 0, maxChunksPerFile: 100 };

describe("chunkMarkdown", () => {
  it("splits on headings and keeps section titles with accurate line numbers", () => {
    const md = [
      "# Guide",
      "",
      "Intro paragraph about setting up payments.",
      "Second line of intro.",
      "",
      "## Retries",
      "",
      "Retries use exponential backoff with a cap of 5 attempts.",
      "",
      "### Temporal",
      "",
      "Temporal gives free shipping for 30 days.",
    ].join("\n");
    const chunks = chunkMarkdown(md, small);
    expect(chunks.some((c) => c.title === "Guide")).toBe(true);
    expect(chunks.some((c) => c.title === "Retries")).toBe(true);
    expect(chunks.some((c) => c.title === "Temporal")).toBe(true);

    const retries = chunks.find((c) => c.title === "Retries")!;
    expect(retries.text).toContain("exponential backoff");
    expect(retries.startLine).toBe(6); // the "## Retries" heading line
    expect(retries.endLine).toBeGreaterThanOrEqual(8); // ...through its paragraph
    expect(retries.text).not.toContain("free shipping for 30");
  });

  it("keeps the nearest preceding heading as chunk title when sections merge", () => {
    const md = [
      "# Big section",
      "",
      Array.from({ length: 6 }, (_, i) => `Sentence number ${i + 1} here.`).join(" "),
      "",
      "## Smaller section",
      "",
      "Short paragraph in the smaller section.",
    ].join("\n");
    const chunks = chunkMarkdown(md, { ...small, chunkSize: 200 });
    const last = chunks[chunks.length - 1]!;
    expect(last.title).toBe("Smaller section");
    expect(last.text).toContain("Short paragraph");
  });

  it("carries overlap across chunk boundaries so context is preserved", () => {
    const para = Array.from({ length: 40 }, (_, i) => `Sentence ${i + 1} of the doc.`).join(" ");
    const md = `# Big\n\n${para}\n\n${para}`;
    const chunks = chunkMarkdown(md, { chunkSize: 200, chunkOverlap: 80, maxChunksPerFile: 100 });
    expect(chunks.length).toBeGreaterThan(1);
    const overlaps = chunks.slice(1).filter((c, i) => {
      const prev = chunks[i]!;
      const prevWords = prev.text.split(" ").slice(-5).join(" ");
      return c.text.includes(prevWords.slice(0, Math.max(10, prevWords.length - 2)));
    });
    expect(overlaps.length).toBeGreaterThan(0);
  });

  it("keeps frontmatter as its own titled chunk", () => {
    const chunks = chunkMarkdown("Body text after frontmatter.\n", opts, "title: My Guide\ncategory: api");
    expect(chunks.some((c) => c.title !== undefined && c.text.includes("title: My Guide"))).toBe(true);
  });
});

describe("chunkRst", () => {
  it("detects underlined section titles like classic .rst", () => {
    const rst = [
      "User Guide",
      "==========",
      "",
      "Welcome paragraph explains installation.",
      "",
      "Webhooks",
      "--------",
      "",
      "Webhooks fire on every payment event.",
    ].join("\n");
    const chunks = chunkRst(rst, small);
    expect(chunks.some((c) => c.title === "User Guide")).toBe(true);
    const webhooks = chunks.find((c) => c.title === "Webhooks");
    expect(webhooks).toBeDefined();
    expect(webhooks!.text).toContain("payment event");
  });
});

describe("chunkText", () => {
  it("splits paragraphs and records 1-based line ranges", () => {
    const txt = "First paragraph body.\n\nSecond paragraph body here.\nWith two lines.";
    const chunks = chunkText(txt, { ...opts, chunkSize: 50, chunkOverlap: 0 });
    expect(chunks).toHaveLength(2);
    expect(chunks[0]!.startLine).toBe(1);
    expect(chunks[0]!.endLine).toBe(1);
    expect(chunks[1]!.startLine).toBe(3);
    expect(chunks[1]!.endLine).toBe(4);
  });

  it("packs small paragraphs into one chunk up to chunkSize", () => {
    const txt = "One.\n\nTwo.\n\nThree.";
    const chunks = chunkText(txt, opts);
    expect(chunks).toHaveLength(1);
    expect(chunks[0]!.text).toContain("One.");
    expect(chunks[0]!.endLine).toBe(5);
  });
});

describe("chunkCode", () => {
  it("keeps doc comments attached and tracks lines", () => {
    const code = [
      "function helper() {}",
      "",
      "/**",
      " * Retries failed webhooks with exponential backoff.",
      " */",
      "function withBackoff() {",
      "  return 1;",
      "}",
    ].join("\n");
    const chunks = chunkCode(code, { ...small, chunkSize: 60, chunkOverlap: 0 });
    const docChunk = chunks.find((c) => c.text.startsWith("/**"))!;
    expect(docChunk).toBeDefined();
    expect(docChunk.startLine).toBe(3);
    expect(docChunk.endLine).toBe(5);
    expect(docChunk.text).toContain("exponential backoff");
  });
});

describe("chunkExtracted (pdf pages)", () => {
  it("splits per page and stamps page numbers", () => {
    const chunks = chunkExtracted(
      "pdf",
      {
        text: "",
        pages: [{ page: 1, text: "Page one content about billing." }, { page: 2, text: "Page two about refunds." }],
      },
      opts,
    );
    expect(chunks.filter((c) => c.page === 1).length).toBe(1);
    expect(chunks.filter((c) => c.page === 2).length).toBe(1);
  });
});
