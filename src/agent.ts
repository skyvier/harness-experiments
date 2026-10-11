import { randomBytes } from "node:crypto";
import type { BaseChatModel } from "@langchain/core/language_models/chat_models";
import { AIMessage, HumanMessage, ToolMessage } from "@langchain/core/messages";
import type { BaseMessage } from "@langchain/core/messages";
import { createAgent, createMiddleware } from "langchain";
import { z } from "zod";

import { calculatorAgentTool } from "./calculator.js";
import { createToolRegistry } from "./tool.js";
import type { ToolFailure, ToolRegistry } from "./tool.js";
import { weatherAgentTool } from "./weather.js";

const SYSTEM_PROMPT = [
  "You are a concise, helpful assistant.",
  "Use the calculate tool for arithmetic.",
  "Use the get_weather tool for current weather when the user provides coordinates in Finland.",
  "If a tool fails, correct the call when possible; otherwise explain the failure.",
  "State uncertainty instead of inventing facts.",
].join(" ");
const MAX_STEPS = 5;
const promptSchema = z.string().trim().min(1);
const textContentSchema = z.union([
  z.string(),
  z.array(
    z.object({ type: z.string(), text: z.string().optional() }).passthrough(),
  ),
]);
const modelResponseSchema = z.object({
  content: textContentSchema,
  tool_calls: z.array(
    z.object({
      id: z.string().min(1),
      name: z.string().min(1),
      args: z.record(z.string(), z.unknown()),
    }),
  ),
  invalid_tool_calls: z.array(
    z.object({
      id: z.string().min(1),
      name: z.string().min(1),
      args: z.string().nullable().optional(),
    }),
  ),
});
const DEFAULT_TOOL_REGISTRY = createToolRegistry([
  calculatorAgentTool,
  weatherAgentTool,
]);

/** An attempted tool call exposed for tracing and diagnostics. */
export interface ToolCallEvent {
  readonly name: string;
  readonly arguments: unknown;
  readonly result: string;
  readonly error?: ToolFailure;
}

/** Optional controls applied to every turn in an agent session. */
export interface AgentOptions {
  readonly maxSteps?: number;
  readonly onToolCall?: (event: ToolCallEvent) => void;
  readonly toolRegistry?: ToolRegistry;
}

/** LangChain model boundary; tests can supply a local BaseChatModel. */
export type AgentModel = BaseChatModel;

/** An in-memory conversation that commits only successfully completed turns. */
export class AgentSession {
  readonly #model: AgentModel;
  readonly #options: AgentOptions;
  #messages: BaseMessage[] = [];
  #turnInProgress = false;

  constructor(model: AgentModel, options: AgentOptions = {}) {
    this.#model = model;
    this.#options = {
      ...options,
      maxSteps: z
        .number()
        .int()
        .positive()
        .parse(options.maxSteps ?? MAX_STEPS),
    };
  }

  async send(prompt: string): Promise<string> {
    if (this.#turnInProgress) {
      throw new Error("An agent turn is already in progress.");
    }
    this.#turnInProgress = true;
    try {
      const messages = [
        ...this.#messages,
        new HumanMessage(promptSchema.parse(prompt)),
      ];
      const {
        maxSteps = MAX_STEPS,
        onToolCall,
        toolRegistry = DEFAULT_TOOL_REGISTRY,
      } = this.#options;
      let modelCalls = 0;
      // LangGraph may schedule tools together. Preserve sequential execution.
      let toolQueue: Promise<unknown> = Promise.resolve();
      const rawArguments = new Map<string, unknown>();
      const usedCallIds = new Set(
        this.#messages.flatMap((message) =>
          ToolMessage.isInstance(message) ? [message.tool_call_id] : [],
        ),
      );
      let fatalToolError: Error | undefined;
      const registerCall = (
        call: { id: string; name: string; args: Record<string, unknown> },
        arguments_: unknown,
      ) => {
        let id = call.id;
        // LangGraph skips calls whose IDs already have a result in history.
        // Preserve execution when a provider reuses an opaque call ID. Mistral
        // accepts nine alphanumeric characters for replacement IDs.
        while (usedCallIds.has(id))
          id = randomBytes(5).toString("hex").slice(0, 9);
        usedCallIds.add(id);
        rawArguments.set(id, arguments_);
        return { ...call, id };
      };
      const middleware = createMiddleware({
        name: "HarnessBoundaries",
        wrapModelCall: async (request, handler) => {
          if (fatalToolError) throw fatalToolError;
          if (modelCalls >= maxSteps) {
            throw new Error(`Agent exceeded its ${maxSteps}-step limit.`);
          }
          modelCalls += 1;
          const response = await handler({ ...request, toolChoice: "auto" });
          if (!AIMessage.isInstance(response)) {
            throw new Error("The model returned no assistant message.");
          }
          const parsed = modelResponseSchema.parse(response);
          // LangChain separates malformed JSON calls from executable calls.
          // Route them through our registry so the model gets a safe failure
          // and can correct the call, rather than silently ending the turn.
          return new AIMessage({
            content: parsed.content,
            tool_calls: [
              ...parsed.tool_calls.map((call) => registerCall(call, call.args)),
              ...parsed.invalid_tool_calls.map((call) =>
                registerCall(
                  { id: call.id, name: call.name, args: {} },
                  call.args,
                ),
              ),
            ],
          });
        },
        wrapToolCall: (request) => {
          const execution = toolQueue.then(async () => {
            const { name, id } = request.toolCall;
            if (!id)
              throw new Error("The model returned a tool call without an id.");
            const arguments_ = rawArguments.get(id);
            const outcome = await toolRegistry.execute(name, arguments_);
            const error = outcome.ok ? undefined : outcome.failure;
            const result = outcome.ok
              ? outcome.result
              : JSON.stringify({ error });
            onToolCall?.({
              name,
              arguments: arguments_,
              result,
              ...(error === undefined ? {} : { error }),
            });
            return new ToolMessage({ name, tool_call_id: id, content: result });
          });
          const guarded = execution.catch((error: unknown) => {
            // ToolNode converts thrown errors into model-visible messages.
            // Abort before the next model call, and never let it expose an
            // observer failure or an unexpected registry exception verbatim.
            fatalToolError =
              error instanceof Error
                ? error
                : new Error("Tool execution failed.");
            throw new Error("Tool execution failed.");
          });
          toolQueue = guarded;
          return guarded;
        },
      });
      const agent = createAgent({
        model: this.#model,
        systemPrompt: SYSTEM_PROMPT,
        tools: toolRegistry.definitions.map((definition) => ({
          ...definition,
        })),
        middleware: [middleware],
      });
      // Graph steps include middleware and tool nodes, so count model calls
      // explicitly above rather than treating recursionLimit as maxSteps.
      const result = await agent.invoke(
        { messages },
        { recursionLimit: maxSteps * 6 + 10 },
      );
      const lastMessage = result.messages.at(-1);
      if (
        !AIMessage.isInstance(lastMessage) ||
        lastMessage.tool_calls?.length
      ) {
        throw new Error("The model returned no final answer.");
      }
      const answer = extractText(lastMessage.content);
      this.#messages = result.messages;
      return answer;
    } finally {
      this.#turnInProgress = false;
    }
  }
}

/** Runs one isolated turn through LangChain's agent loop. */
export async function runAgent(
  prompt: string,
  model: AgentModel,
  options: AgentOptions | number = {},
): Promise<string> {
  return new AgentSession(
    model,
    typeof options === "number" ? { maxSteps: options } : options,
  ).send(prompt);
}

/** Validates and extracts displayable text from model content. */
export function extractText(content: unknown): string {
  const parsed = textContentSchema.safeParse(content);
  const text = parsed.success
    ? typeof parsed.data === "string"
      ? parsed.data
      : parsed.data
          .filter((chunk) => chunk.type === "text")
          .map((chunk) => chunk.text ?? "")
          .join("")
    : "";
  if (!text.trim()) throw new Error("Mistral returned no text content.");
  return text;
}
