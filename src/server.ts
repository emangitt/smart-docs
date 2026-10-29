import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { VERSION } from "./version.js";
import { loadConfig } from "./config.js";
import { Services } from "./services.js";
import { defineTools } from "./tools.js";

/** Create a configured MCP server instance (does not connect a transport). */
export function createSmartDocsServer(services: Services): McpServer {
  const server = new McpServer(
    { name: "smart-docs", version: VERSION },
    {
      instructions:
        "smart-docs indexes local documentation (md/txt/rst/pdf/code) into an offline vector index. " +
        "First call index_directory with a folder path, then semantic_search / ask_docs / get_file_summary to ground answers in it. " +
        "list_indexed shows state; clear_index (confirm: true) resets. All data stays on this machine.",
    },
  );

  for (const tool of defineTools(services)) {
    server.registerTool(tool.name, {
      title: tool.title,
      description: tool.description,
      inputSchema: tool.inputSchema,
      annotations: tool.annotations ?? {},
    }, tool.handler as Parameters<typeof server.registerTool>[2]);
  }
  return server;
}

/** CLI entry point: stdio transport, logging to stderr only. */
export async function main(argv: string[] = process.argv.slice(2)): Promise<void> {
  let services: Services;
  try {
    const cfg = loadConfig(argv);
    services = new Services(cfg);
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    process.stderr.write(`smart-docs: ${msg}\n`);
    process.exitCode = 1;
    return;
  }

  // Keep stdio clean: everything else must go to stderr.
  process.stderr.write(`smart-docs v${VERSION} — index dir: ${services.indexDir}\n`);
  if (services.cfg.embeddingProvider === "remote") {
    process.stderr.write("smart-docs: using REMOTE embeddings (calls require network)\n");
  } else {
    process.stderr.write(
      `smart-docs: local embeddings (${services.cfg.embeddingModel}); first run downloads and caches the model\n`,
    );
  }

  const server = createSmartDocsServer(services);
  const transport = new StdioServerTransport();
  await server.connect(transport);
}
