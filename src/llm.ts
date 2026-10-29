import { SmartDocsError } from "./errors.js";
import type { Config, LlmConfig } from "./config.js";
import type { SummaryPassage } from "./summarize.js";

export interface LlmClient {
  chat(system: string, user: string): Promise<string>;
}

/** OpenAI-compatible /chat/completions client (OpenAI, OpenRouter, Ollama, LM Studio, ...). */
export class RemoteLlmClient implements LlmClient {
  constructor(private readonly cfg: LlmConfig) {}

  async chat(system: string, user: string): Promise<string> {
    const headers: Record<string, string> = { "content-type": "application/json" };
    if (this.cfg.apiKey) headers.authorization = `Bearer ${this.cfg.apiKey}`;
    let res: Response;
    try {
      res = await fetch(`${this.cfg.baseUrl.replace(/\/+$/, "")}/chat/completions`, {
        method: "POST",
        headers,
        body: JSON.stringify({
          model: this.cfg.model,
          messages: [
            { role: "system", content: system },
            { role: "user", content: user },
          ],
          temperature: 0.2,
        }),
        signal: AbortSignal.timeout(120_000),
      });
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      throw new SmartDocsError(
        `LLM request to ${this.cfg.baseUrl} failed: ${msg}`,
        "Check SMART_DOCS_LLM_* settings, or unset them to use offline extractive summaries.",
      );
    }
    if (!res.ok) {
      const body = await res.text().catch(() => "");
      throw new SmartDocsError(
        `LLM returned HTTP ${res.status} ${body.slice(0, 300)}`,
        "Check the model name and API key; or unset SMART_DOCS_LLM_* to fall back to extractive mode.",
      );
    }
    const json = (await res.json()) as { choices?: Array<{ message?: { content?: string } }> };
    const content = json.choices?.[0]?.message?.content;
    if (!content) throw new SmartDocsError("LLM returned no content.", "Try a different SMART_DOCS_LLM_MODEL.");
    return content;
  }
}

export function getLlm(cfg: Config): LlmClient | null {
  return cfg.llm ? new RemoteLlmClient(cfg.llm) : null;
}

const ASK_SYSTEM = `You answer questions strictly from the provided context passages from the user's local documentation.
Rules:
- Use ONLY the numbered CONTEXT passages. Do not invent facts or use outside knowledge.
- Cite passages inline as [1], [2] ... matching the passage numbers.
- If the context is insufficient, say exactly what is missing and suggest what file to look at or index.`;

const SUMMARIZE_SYSTEM = `You summarize documentation/source files from the provided context passages.
Rules:
- Use ONLY the passages; do not invent content.
- Cite passages inline as [1], [2] ...
- Structure: 2-4 sentence overview, then short bullet highlights. Keep it under ~250 words.`;

function passageBlock(items: SummaryPassage[]): string {
  return items
    .map((p, i) => {
      const loc = p.page !== undefined ? `page ${p.page}` : `lines ${p.startLine}-${p.endLine}`;
      const head = p.title ? ` — ${p.title}` : "";
      return `[${i + 1}] ${p.path}${loc && head ? "" : ""} (${loc})${head}\n${p.text.slice(0, 2500)}`;
    })
    .join("\n\n");
}

export async function askWithLlm(
  llm: LlmClient,
  question: string,
  passages: SummaryPassage[],
): Promise<string> {
  const user = `CONTEXT:\n\n${passageBlock(passages)}\n\nQUESTION: ${question}`;
  return llm.chat(ASK_SYSTEM, user);
}

export async function summarizeWithLlm(llm: LlmClient, label: string, passages: SummaryPassage[]): Promise<string> {
  const user = `CONTEXT:\n\n${passageBlock(passages)}\n\nSummarize this ${label}.`;
  return llm.chat(SUMMARIZE_SYSTEM, user);
}
