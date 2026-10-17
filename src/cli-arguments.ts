import { z } from "zod";

import { parseThreadId } from "./persistence.js";

const argumentsSchema = z.array(z.string()).max(256);
const promptSchema = z.string().trim().min(1);

export type CliCommand =
  | { readonly mode: "prompt"; readonly prompt: string }
  | { readonly mode: "chat" }
  | { readonly mode: "durable-chat"; readonly threadId: string };

/** Validates process arguments and selects one supported CLI mode. */
export function parseCliArguments(args: unknown): CliCommand {
  const parsed = argumentsSchema.parse(args);

  if (parsed.length === 1 && parsed[0] === "--chat") {
    return { mode: "chat" };
  }
  if (
    parsed.length === 3 &&
    parsed[0] === "--chat" &&
    parsed[1] === "--thread"
  ) {
    return { mode: "durable-chat", threadId: parseThreadId(parsed[2]) };
  }
  if (parsed[0] === "--chat" || parsed.includes("--thread")) {
    throw new Error(cliUsage());
  }

  const prompt = promptSchema.safeParse(parsed.join(" "));
  if (!prompt.success) throw new Error(cliUsage());
  return { mode: "prompt", prompt: prompt.data };
}

export function cliUsage(): string {
  return 'Usage: npm start -- "your prompt", npm start -- --chat, or npm start -- --chat --thread THREAD_ID';
}
