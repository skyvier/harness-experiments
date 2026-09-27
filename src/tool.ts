import type {
  ChatCompletionRequestTool,
  FunctionTool,
} from "@mistralai/mistralai/models/components";
import { z } from "zod";

type ToolResult = string | Promise<string>;

const toolFailureSchema = z
  .object({
    code: z.string().regex(/^[a-z][a-z0-9_]*$/).max(64),
    message: z.string().min(1).max(200),
    retryable: z.boolean(),
  })
  .strict();

/** Model-safe information about a failed tool execution. */
export type ToolFailure = Readonly<z.infer<typeof toolFailureSchema>>;

const GENERIC_TOOL_FAILURE: ToolFailure = Object.freeze({
  code: "tool_error",
  message: "The tool encountered an error.",
  retryable: false,
});

/** The result of executing a tool across the untrusted model boundary. */
export type ToolExecutionResult =
  | { readonly ok: true; readonly result: string }
  | { readonly ok: false; readonly failure: ToolFailure };

/** A function tool with validation hidden behind an untyped execution boundary. */
export interface AgentTool<
  Name extends string = string,
  Result extends ToolResult = ToolResult,
> {
  readonly name: Name;
  readonly definition: FunctionTool;
  /** Maps recognized errors to code-owned text that is safe for model context. */
  readonly mapError?: (error: unknown) => ToolFailure | undefined;
  execute(arguments_: unknown): Result;
}

interface ToolConfiguration<
  Name extends string,
  Schema extends z.ZodType,
  Result extends ToolResult,
> {
  readonly name: Name;
  readonly description: string;
  readonly argumentsSchema: Schema;
  /** Maps recognized errors to code-owned text that is safe for model context. */
  readonly mapError?: (error: unknown) => ToolFailure | undefined;
  readonly execute: (arguments_: z.output<Schema>) => Result;
}

/** Creates a tool whose executor can only receive schema-validated arguments. */
export function defineTool<
  const Name extends string,
  Schema extends z.ZodType,
  Result extends ToolResult,
>(
  configuration: ToolConfiguration<Name, Schema, Result>,
): AgentTool<Name, Result> {
  const { name, description, argumentsSchema, execute, mapError } =
    configuration;

  return {
    name,
    ...(mapError === undefined ? {} : { mapError }),
    definition: {
      type: "function",
      function: {
        name,
        description,
        strict: true,
        parameters: z.toJSONSchema(argumentsSchema),
      },
    },
    execute(arguments_: unknown): Result {
      const parsed = argumentsSchema.parse(parseArguments(name, arguments_));
      return execute(parsed);
    },
  };
}

/** A closed collection of tool definitions and their matching executors. */
export interface ToolRegistry {
  readonly definitions: readonly ChatCompletionRequestTool[];
  execute(name: string, arguments_: unknown): Promise<ToolExecutionResult>;
}

/** Creates a registry and rejects ambiguous duplicate tool names. */
export function createToolRegistry(
  tools: readonly AgentTool[],
): ToolRegistry {
  const toolsByName = new Map<string, AgentTool>();

  for (const tool of tools) {
    if (tool.definition.function.name !== tool.name) {
      throw new Error(`Tool name mismatch: ${tool.name}`);
    }
    if (toolsByName.has(tool.name)) {
      throw new Error(`Duplicate tool: ${tool.name}`);
    }
    toolsByName.set(tool.name, tool);
  }

  return {
    definitions: Object.freeze(tools.map((tool) => tool.definition)),
    async execute(
      name: string,
      arguments_: unknown,
    ): Promise<ToolExecutionResult> {
      const tool = toolsByName.get(name);
      if (!tool) {
        return { ok: false, failure: GENERIC_TOOL_FAILURE };
      }

      try {
        return { ok: true, result: await tool.execute(arguments_) };
      } catch (error: unknown) {
        return { ok: false, failure: mapToolFailure(tool, error) };
      }
    },
  };
}

/** Validates a tool-owned allowlist mapping and falls back safely. */
function mapToolFailure(tool: AgentTool, error: unknown): ToolFailure {
  try {
    const mapped = tool.mapError?.(error);
    const parsed = toolFailureSchema.safeParse(mapped);
    return parsed.success ? Object.freeze(parsed.data) : GENERIC_TOOL_FAILURE;
  } catch {
    return GENERIC_TOOL_FAILURE;
  }
}

function parseArguments(toolName: string, arguments_: unknown): unknown {
  if (typeof arguments_ !== "string") {
    return arguments_;
  }

  try {
    return JSON.parse(arguments_);
  } catch (error: unknown) {
    throw new Error(`Tool ${toolName} arguments are not valid JSON.`, {
      cause: error,
    });
  }
}
