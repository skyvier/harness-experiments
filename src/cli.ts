import { existsSync } from "node:fs";
import { loadEnvFile } from "node:process";

import { readConfig } from "./config.js";
import { createAgentSession, generateText } from "./harness.js";
import { runInteractiveChat } from "./interactive-chat.js";

/** Runs interactive chat or a single prompt supplied as command-line arguments. */
async function main(args: readonly string[]): Promise<void> {
  if (existsSync(".env")) {
    loadEnvFile(".env");
  }

  if (args.length === 1 && args[0] === "--chat") {
    await runInteractiveChat(
      createAgentSession(readConfig()),
      process.stdin,
      process.stdout,
      process.stderr,
    );
    return;
  }

  const prompt = args.join(" ").trim();
  if (!prompt) {
    throw new Error('Usage: npm start -- "your prompt" or npm start -- --chat');
  }

  const answer = await generateText(prompt, readConfig());
  process.stdout.write(`${answer}\n`);
}

main(process.argv.slice(2)).catch((error: unknown) => {
  const message = error instanceof Error ? error.message : String(error);
  process.stderr.write(`Error: ${message}\n`);
  process.exitCode = 1;
});
