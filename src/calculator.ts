import type { ChatCompletionRequestTool } from "@mistralai/mistralai/models/components";
import { z } from "zod";

const calculatorArgumentsSchema = z
  .object({
    operation: z.enum(["add", "subtract", "multiply", "divide"]),
    left: z.number().finite(),
    right: z.number().finite(),
  })
  .strict()
  .superRefine((arguments_, context) => {
    if (arguments_.operation === "divide" && arguments_.right === 0) {
      context.addIssue({
        code: "custom",
        message: "Cannot divide by zero",
        path: ["right"],
      });
    }
  });

export const calculatorTool = {
  type: "function",
  function: {
    name: "calculate",
    description: "Perform arithmetic on two numbers.",
    strict: true,
    parameters: z.toJSONSchema(calculatorArgumentsSchema),
  },
} satisfies ChatCompletionRequestTool;

/** Validates model-provided arguments and executes the calculator tool. */
export function executeCalculator(arguments_: unknown): string {
  const parsed = calculatorArgumentsSchema.parse(parseArguments(arguments_));
  const { operation, left, right } = parsed;

  const result = (() => {
    switch (operation) {
      case "add":
        return left + right;
      case "subtract":
        return left - right;
      case "multiply":
        return left * right;
      case "divide":
        return left / right;
    }
  })();

  return JSON.stringify({ result });
}

function parseArguments(arguments_: unknown): unknown {
  if (typeof arguments_ !== "string") {
    return arguments_;
  }

  try {
    return JSON.parse(arguments_);
  } catch (error: unknown) {
    throw new Error("Calculator arguments are not valid JSON.", { cause: error });
  }
}
