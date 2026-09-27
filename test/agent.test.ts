import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { z } from "zod";

import { runAgent } from "../src/agent.js";
import type { AgentModel, ToolCallEvent } from "../src/agent.js";
import { createToolRegistry, defineTool } from "../src/tool.js";

describe("runAgent", () => {
  it("executes a calculator call and returns the final answer", async () => {
    const requests: Parameters<AgentModel["complete"]>[0][] = [];
    const toolCalls: ToolCallEvent[] = [];
    const responses = [
      {
        toolCalls: [
          {
            id: "call-1",
            function: {
              name: "calculate",
              arguments: {
                operation: "multiply",
                left: 137,
                right: 42,
              },
            },
          },
        ],
      },
      { content: "137 multiplied by 42 is 5754." },
    ];
    const model: AgentModel = {
      async complete(messages) {
        requests.push(messages);
        const response = responses.shift();
        assert.ok(response);
        return response;
      },
    };

    const result = await runAgent("What is 137 * 42?", model, {
      onToolCall: (event) => toolCalls.push(event),
    });

    assert.equal(result, "137 multiplied by 42 is 5754.");
    assert.equal(requests.length, 2);
    assert.deepEqual(requests[1]?.at(-1), {
      role: "tool",
      name: "calculate",
      toolCallId: "call-1",
      content: '{"result":5754}',
    });
    assert.deepEqual(toolCalls, [
      {
        name: "calculate",
        arguments: {
          operation: "multiply",
          left: 137,
          right: 42,
        },
        result: '{"result":5754}',
      },
    ]);
  });

  it("returns invalid tool arguments to the model so it can retry", async () => {
    const requests: Parameters<AgentModel["complete"]>[0][] = [];
    const toolCalls: ToolCallEvent[] = [];
    const responses = [
      {
        toolCalls: [
          {
            id: "call-1",
            function: {
              name: "calculate",
              arguments: {
                operation: "divide",
                left: 1,
                right: 0,
              },
            },
          },
        ],
      },
      {
        toolCalls: [
          {
            id: "call-2",
            function: {
              name: "calculate",
              arguments: {
                operation: "divide",
                left: 1,
                right: 2,
              },
            },
          },
        ],
      },
      { content: "One divided by two is 0.5." },
    ];
    const model: AgentModel = {
      async complete(messages) {
        requests.push([...messages]);
        const response = responses.shift();
        assert.ok(response);
        return response;
      },
    };

    const result = await runAgent("Divide one by zero", model, {
      onToolCall: (event) => toolCalls.push(event),
    });

    assert.equal(result, "One divided by two is 0.5.");
    assert.match(
      String(requests[1]?.at(-1)?.content),
      /Cannot divide by zero/,
    );
    assert.deepEqual(requests[2]?.at(-1), {
      role: "tool",
      name: "calculate",
      toolCallId: "call-2",
      content: '{"result":0.5}',
    });
    assert.equal(toolCalls.length, 2);
    assert.match(toolCalls[0]?.error ?? "", /Cannot divide by zero/);
  });

  it("returns executor failures to the model", async () => {
    const failingTool = defineTool({
      name: "fail",
      description: "Always fail.",
      argumentsSchema: z.object({}).strict(),
      execute: () => {
        throw new Error("Service unavailable");
      },
    });
    const requests: Parameters<AgentModel["complete"]>[0][] = [];
    const responses = [
      {
        toolCalls: [
          {
            id: "call-1",
            function: { name: "fail", arguments: {} },
          },
        ],
      },
      { content: "The service is unavailable." },
    ];
    const model: AgentModel = {
      async complete(messages) {
        requests.push([...messages]);
        const response = responses.shift();
        assert.ok(response);
        return response;
      },
    };

    const result = await runAgent("Use the service", model, {
      toolRegistry: createToolRegistry([failingTool]),
    });

    assert.equal(result, "The service is unavailable.");
    assert.deepEqual(requests[1]?.at(-1), {
      role: "tool",
      name: "fail",
      toolCallId: "call-1",
      content: '{"error":"Service unavailable"}',
    });
  });

  it("stops a model that never produces a final answer", async () => {
    const model: AgentModel = {
      async complete() {
        return {
          toolCalls: [
            {
              id: "call-1",
              function: {
                name: "calculate",
                arguments: {
                  operation: "add",
                  left: 1,
                  right: 1,
                },
              },
            },
          ],
        };
      },
    };

    await assert.rejects(
      () => runAgent("Keep calculating", model, { maxSteps: 2 }),
      /2-step limit/,
    );
  });
});
