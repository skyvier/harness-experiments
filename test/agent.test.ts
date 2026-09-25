import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { runAgent } from "../src/agent.js";
import type { AgentModel } from "../src/agent.js";

describe("runAgent", () => {
  it("executes a calculator call and returns the final answer", async () => {
    const requests: Parameters<AgentModel["complete"]>[0][] = [];
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

    const result = await runAgent("What is 137 * 42?", model);

    assert.equal(result, "137 multiplied by 42 is 5754.");
    assert.equal(requests.length, 2);
    assert.deepEqual(requests[1]?.at(-1), {
      role: "tool",
      name: "calculate",
      toolCallId: "call-1",
      content: '{"result":5754}',
    });
  });

  it("rejects invalid tool arguments", async () => {
    const model: AgentModel = {
      async complete() {
        return {
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
        };
      },
    };

    await assert.rejects(() => runAgent("Divide by zero", model));
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
      () => runAgent("Keep calculating", model, 2),
      /2-step limit/,
    );
  });
});
