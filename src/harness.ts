import { Mistral } from "@mistralai/mistralai";

import type { HarnessConfig } from "./config.js";

const SYSTEM_PROMPT =
  "You are a concise, helpful assistant. State uncertainty instead of inventing facts.";

/** Sends one prompt to Mistral's EU-hosted inference endpoint. */
export async function generateText(
  prompt: string,
  config: HarnessConfig,
): Promise<string> {
  const client = new Mistral({
    apiKey: config.apiKey,
    server: "eu",
  });

  const response = await client.chat.complete({
    model: config.model,
    messages: [
      { role: "system", content: SYSTEM_PROMPT },
      { role: "user", content: prompt },
    ],
  });

  return extractText(response?.choices?.[0]?.message?.content);
}

/** Normalizes the SDK's string-or-content-chunks response into displayable text. */
export function extractText(content: unknown): string {
  const text =
    typeof content === "string"
      ? content
      : Array.isArray(content)
        ? content
            .filter(isTextChunk)
            .map((chunk) => chunk.text)
            .join("")
        : "";

  if (!text.trim()) {
    throw new Error("Mistral returned no text content.");
  }

  return text;
}

function isTextChunk(value: unknown): value is { readonly text: string } {
  return (
    typeof value === "object" &&
    value !== null &&
    "text" in value &&
    typeof value.text === "string"
  );
}
