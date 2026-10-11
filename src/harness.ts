import { ChatMistralAI } from "@langchain/mistralai";
import { z } from "zod";

import { AgentSession } from "./agent.js";
import type { HarnessConfig } from "./config.js";

/** Creates the LangChain provider adapter with the existing inference policy. */
export function createMistralModel(config: HarnessConfig): ChatMistralAI {
  return new ChatMistralAI({
    apiKey: config.apiKey,
    model: config.model,
    serverURL: "https://api.eu.mistral.ai",
    maxRetries: 0,
    beforeRequestHooks: [
      async (request) => {
        // ChatMistralAI currently omits parallelToolCalls from its public options.
        // Preserve our provider setting at the SDK's supported request hook.
        const payload = z
          .record(z.string(), z.unknown())
          .parse(await request.clone().json());
        return new Request(request, {
          body: JSON.stringify({ ...payload, parallel_tool_calls: false }),
        });
      },
    ],
  });
}

/** Runs an isolated turn using Mistral's EU-hosted inference endpoint. */
export async function generateText(
  prompt: string,
  config: HarnessConfig,
): Promise<string> {
  return createAgentSession(config).send(prompt);
}

/** Creates a stateful agent session backed by LangChain and Mistral. */
export function createAgentSession(config: HarnessConfig): AgentSession {
  return new AgentSession(createMistralModel(config), {
    onToolCall: ({ name, arguments: arguments_, result, error }) => {
      const outcome =
        error === undefined
          ? `-> ${result}`
          : `failed [${error.code}]: ${error.message}`;
      process.stderr.write(
        `[tool] ${name} ${JSON.stringify(arguments_)} ${outcome}\n`,
      );
    },
  });
}
