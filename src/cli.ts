import { existsSync } from "node:fs";
import { loadEnvFile } from "node:process";

import { readConfig } from "./config.js";
import { generateText } from "./harness.js";

/** Runs a single prompt supplied as command-line arguments. */
async function main(args: readonly string[]): Promise<void> {
  if (existsSync(".env")) {
    loadEnvFile(".env");
  }

  const prompt = args.join(" ").trim();
  if (!prompt) {
    throw new Error('Usage: npm start -- "your prompt"');
  }

  const answer = await generateText(prompt, readConfig());
  process.stdout.write(`${answer}\n`);
}

main(process.argv.slice(2)).catch((error: unknown) => {
  const message = error instanceof Error ? error.message : String(error);
  process.stderr.write(`Error: ${message}\n`);
  process.exitCode = 1;
});
