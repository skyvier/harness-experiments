import { Mistral } from "@mistralai/mistralai";

import { runAgent } from "./agent.js";
import type { AgentModel } from "./agent.js";
import type { HarnessConfig } from "./config.js";

/** Runs the agent loop using Mistral's EU-hosted inference endpoint. */
export async function generateText(
  prompt: string,
  config: HarnessConfig,
): Promise<string> {
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

  return runAgent(prompt, model, {
    onToolCall: ({ name, arguments: arguments_, result }) => {
      process.stderr.write(
        `[tool] ${name} ${JSON.stringify(arguments_)} -> ${result}\n`,
      );
    },
  });
}
