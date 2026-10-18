import { Mistral } from "@mistralai/mistralai";

import { AgentSession } from "./agent.js";
import type { AgentModel } from "./agent.js";
import type { HarnessConfig } from "./config.js";

/** Runs the agent loop using Mistral's EU-hosted inference endpoint. */
export async function generateText(
  prompt: string,
  config: HarnessConfig,
): Promise<string> {
  return createAgentSession(config).send(prompt);
}

/** Creates a stateful agent session backed by Mistral inference. */
export function createAgentSession(config: HarnessConfig): AgentSession {
  const client = new Mistral({
    apiKey: config.apiKey,
    server: "eu",
  });

  const model: AgentModel = {
    async complete(messages, tools) {
      const response = await client.chat.complete({
        model: config.model,
        messages: [...messages],
        tools: [...tools],
        toolChoice: "auto",
        parallelToolCalls: false,
      });
      const message = response?.choices?.[0]?.message;

      if (!message) {
        throw new Error("Mistral returned no assistant message.");
      }

      return message;
    },
  };

  return new AgentSession(model, {
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
