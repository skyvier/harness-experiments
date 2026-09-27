import type {
  AssistantMessage,
  AssistantMessageContent,
  ChatCompletionRequestMessage,
  ChatCompletionRequestTool,
} from "@mistralai/mistralai/models/components";

import { calculatorAgentTool } from "./calculator.js";
import { createToolRegistry } from "./tool.js";
import type { ToolFailure, ToolRegistry } from "./tool.js";

const SYSTEM_PROMPT = [
  "You are a concise, helpful assistant.",
  "Use the calculate tool for arithmetic.",
  "If a tool fails, correct the call when possible; otherwise explain the failure.",
  "State uncertainty instead of inventing facts.",
].join(" ");
const MAX_STEPS = 5;
const DEFAULT_TOOL_REGISTRY = createToolRegistry([calculatorAgentTool]);

/** An attempted tool call exposed for tracing and diagnostics. */
export interface ToolCallEvent {
  readonly name: string;
  readonly arguments: unknown;
  readonly result: string;
  readonly error?: ToolFailure;
}

/** Optional controls for one agent-loop invocation. */
export interface AgentOptions {
  readonly maxSteps?: number;
  readonly onToolCall?: (event: ToolCallEvent) => void;
  readonly toolRegistry?: ToolRegistry;
}

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
  options: AgentOptions | number = {},
): Promise<string> {
  const normalizedOptions =
    typeof options === "number" ? { maxSteps: options } : options;
  const {
    maxSteps = MAX_STEPS,
    onToolCall,
    toolRegistry = DEFAULT_TOOL_REGISTRY,
  } = normalizedOptions;
  const messages: ChatCompletionRequestMessage[] = [
    { role: "system", content: SYSTEM_PROMPT },
    { role: "user", content: prompt },
  ];

  for (let step = 0; step < maxSteps; step += 1) {
    const response = await model.complete(messages, toolRegistry.definitions);
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

      const execution = await toolRegistry.execute(
        toolCall.function.name,
        toolCall.function.arguments,
      );
      const error = execution.ok ? undefined : execution.failure;
      const result = execution.ok
        ? execution.result
        : JSON.stringify({ error: execution.failure });

      onToolCall?.({
        name: toolCall.function.name,
        arguments: toolCall.function.arguments,
        result,
        ...(error === undefined ? {} : { error }),
      });

      messages.push({
        role: "tool",
        name: toolCall.function.name,
        toolCallId,
        content: result,
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
