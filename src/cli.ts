#!/usr/bin/env node
import { main } from "./server.js";

main().catch((err) => {
  process.stderr.write(`smart-docs: fatal: ${err instanceof Error ? (err.stack ?? err.message) : String(err)}\n`);
  process.exit(1);
});
