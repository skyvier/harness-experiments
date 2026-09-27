import type {
  ChatCompletionRequestTool,
  FunctionTool,
} from "@mistralai/mistralai/models/components";
import { z } from "zod";

type ToolResult = string | Promise<string>;

/** A function tool with validation hidden behind an untyped execution boundary. */
export interface AgentTool<
  Name extends string = string,
  Result extends ToolResult = ToolResult,
> {
  readonly name: Name;
  readonly definition: FunctionTool;
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
  const { name, description, argumentsSchema, execute } = configuration;

  return {
    name,
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
  execute(name: string, arguments_: unknown): Promise<string>;
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
    async execute(name: string, arguments_: unknown): Promise<string> {
      const tool = toolsByName.get(name);
      if (!tool) {
        throw new Error(`Unknown tool: ${name}`);
      }
      return tool.execute(arguments_);
    },
  };
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
