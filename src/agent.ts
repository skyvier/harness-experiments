import type {
  AssistantMessage,
  AssistantMessageContent,
  ChatCompletionRequestMessage,
  ChatCompletionRequestTool,
} from "@mistralai/mistralai/models/components";

import { calculatorTool, executeCalculator } from "./calculator.js";

const SYSTEM_PROMPT = [
  "You are a concise, helpful assistant.",
  "Use the calculate tool for arithmetic.",
  "State uncertainty instead of inventing facts.",
].join(" ");
const MAX_STEPS = 5;

/** Boundary implemented by an LLM provider adapter. */
export interface AgentModel {
  complete(
    messages: readonly ChatCompletionRequestMessage[],
    tools: readonly ChatCompletionRequestTool[],
  ): Promise<AssistantMessage>;
}

/** Runs a bounded model/tool loop until the model produces a final answer. */
export async function runAgent(
  prompt: string,
  model: AgentModel,
  maxSteps = MAX_STEPS,
): Promise<string> {
  const messages: ChatCompletionRequestMessage[] = [
    { role: "system", content: SYSTEM_PROMPT },
    { role: "user", content: prompt },
  ];
  const tools = [calculatorTool];

  for (let step = 0; step < maxSteps; step += 1) {
    const response = await model.complete(messages, tools);
    const toolCalls = response.toolCalls ?? [];

    if (toolCalls.length === 0) {
      return extractText(response.content);
    }

    messages.push({ ...response, role: "assistant" });

    for (const toolCall of toolCalls) {
      const toolCallId = toolCall.id;
      if (!toolCallId) {
        throw new Error("Mistral returned a tool call without an id.");
      }

      if (toolCall.function.name !== "calculate") {
        throw new Error(`Unknown tool: ${toolCall.function.name}`);
      }

      messages.push({
        role: "tool",
        name: toolCall.function.name,
        toolCallId,
        content: executeCalculator(toolCall.function.arguments),
      });
    }
  }

  throw new Error(`Agent exceeded its ${maxSteps}-step limit.`);
}

/** Extracts displayable text from content already validated by the model SDK. */
export function extractText(
  content: AssistantMessageContent | null | undefined,
): string {
  const text =
    typeof content === "string"
      ? content
      : content
          ?.filter((chunk) => chunk.type === "text")
          .map((chunk) => chunk.text)
          .join("") ?? "";

  if (!text.trim()) {
    throw new Error("Mistral returned no text content.");
  }

  return text;
}
