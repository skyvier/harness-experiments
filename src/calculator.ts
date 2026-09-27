import { z } from "zod";

import { defineTool } from "./tool.js";

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
type CalculatorArguments = z.infer<typeof calculatorArgumentsSchema>;

export const calculatorAgentTool = defineTool({
  name: "calculate",
  description: "Perform arithmetic on two numbers.",
  argumentsSchema: calculatorArgumentsSchema,
  execute(arguments_: CalculatorArguments): string {
    return JSON.stringify({ result: calculate(arguments_) });
  },
});

export const calculatorTool = calculatorAgentTool.definition;

/** Validates model-provided arguments and executes the calculator tool. */
export function executeCalculator(arguments_: unknown): string {
  return calculatorAgentTool.execute(arguments_);
}

/** Applies a validated arithmetic operation. */
function calculate({
  operation,
  left,
  right,
}: CalculatorArguments): number {
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
}
