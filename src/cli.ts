import { existsSync } from "node:fs";
import { loadEnvFile } from "node:process";

import { parseCliArguments } from "./cli-arguments.js";
import { readConfig, readPersistenceConfig } from "./config.js";
import {
  createAgentSession,
  createDurableAgentSession,
  generateText,
} from "./harness.js";
import { runInteractiveChat } from "./interactive-chat.js";

/** Runs interactive chat or a single prompt supplied as command-line arguments. */
async function main(args: readonly string[]): Promise<void> {
  if (existsSync(".env")) {
    loadEnvFile(".env");
  }

  const command = parseCliArguments(args);

  if (command.mode === "chat") {
    await runInteractiveChat(
      createAgentSession(readConfig()),
      process.stdin,
      process.stdout,
      process.stderr,
    );
    return;
  }

  if (command.mode === "durable-chat") {
    const durable = await createDurableAgentSession(
      readConfig(),
      command.threadId,
      readPersistenceConfig().databasePath,
    );
    try {
      await runInteractiveChat(
        durable.session,
        process.stdin,
        process.stdout,
        process.stderr,
      );
    } finally {
      durable.close();
    }
    return;
  }

  const answer = await generateText(command.prompt, readConfig());
  process.stdout.write(`${answer}\n`);
}

main(process.argv.slice(2)).catch((error: unknown) => {
  const message = error instanceof Error ? error.message : String(error);
  process.stderr.write(`Error: ${message}\n`);
  process.exitCode = 1;
});
