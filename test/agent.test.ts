import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { AIMessage, ToolMessage } from "@langchain/core/messages";
import { z } from "zod";

import { AgentSession, runAgent } from "../src/agent.js";
import type { ToolCallEvent } from "../src/agent.js";
import { createToolRegistry, defineTool } from "../src/tool.js";
import { ScriptedModel } from "./helpers/model.js";

function call(
  id: string,
  name: string,
  args: Record<string, unknown>,
): AIMessage {
  return new AIMessage({ content: "", tool_calls: [{ id, name, args }] });
}

function scripted(responses: AIMessage[]): ScriptedModel {
  return new ScriptedModel(() => {
    const response = responses.shift();
    assert.ok(response);
    return response;
  });
}

function assertToolResult(
  model: ScriptedModel,
  request: number,
  name: string,
  id: string,
  content: string,
): void {
  const message = model.requests[request]?.at(-1);
  assert.ok(ToolMessage.isInstance(message));
  assert.equal(message.name, name);
  assert.equal(message.tool_call_id, id);
  assert.equal(message.content, content);
}

const genericFailure =
  '{"error":{"code":"tool_error","message":"The tool encountered an error.","retryable":false}}';

describe("runAgent through LangChain", () => {
  it("executes a calculator call and returns the final answer", async () => {
    const args = { operation: "multiply", left: 137, right: 42 };
    const model = scripted([
      call("call-1", "calculate", args),
      new AIMessage("137 multiplied by 42 is 5754."),
    ]);
    const events: ToolCallEvent[] = [];
    assert.equal(
      await runAgent("What is 137 * 42?", model, {
        onToolCall: (event) => events.push(event),
      }),
      "137 multiplied by 42 is 5754.",
    );
    assert.equal(model.requests.length, 2);
    assertToolResult(model, 1, "calculate", "call-1", '{"result":5754}');
    assert.deepEqual(events, [
      { name: "calculate", arguments: args, result: '{"result":5754}' },
    ]);
    assert.equal(model.boundTools.length, 2);
  });

  it("returns invalid tool arguments to the model so it can retry", async () => {
    const model = scripted([
      call("call-1", "calculate", { operation: "divide", left: 1, right: 0 }),
      call("call-2", "calculate", { operation: "divide", left: 1, right: 2 }),
      new AIMessage("One divided by two is 0.5."),
    ]);
    const events: ToolCallEvent[] = [];
    assert.equal(
      await runAgent("Divide one by zero", model, {
        onToolCall: (event) => events.push(event),
      }),
      "One divided by two is 0.5.",
    );
    assertToolResult(model, 1, "calculate", "call-1", genericFailure);
    assertToolResult(model, 2, "calculate", "call-2", '{"result":0.5}');
    assert.equal(events.length, 2);
    assert.equal(events[0]?.error?.code, "tool_error");
  });

  it("keeps upstream errors out of model context and tracing", async () => {
    const failingTool = defineTool({
      name: "fail",
      description: "Always fail.",
      argumentsSchema: z.object({}).strict(),
      execute: () => {
        throw new Error("upstream included private-token-123");
      },
      mapError: () => ({
        code: "service_unavailable",
        message: "The service is temporarily unavailable.",
        retryable: true,
      }),
    });
    const model = scripted([
      call("call-1", "fail", {}),
      new AIMessage("The service is unavailable."),
    ]);
    const events: ToolCallEvent[] = [];
    assert.equal(
      await runAgent("Use the service", model, {
        toolRegistry: createToolRegistry([failingTool]),
        onToolCall: (event) => events.push(event),
      }),
      "The service is unavailable.",
    );
    assertToolResult(
      model,
      1,
      "fail",
      "call-1",
      '{"error":{"code":"service_unavailable","message":"The service is temporarily unavailable.","retryable":true}}',
    );
    assert.doesNotMatch(
      JSON.stringify([model.requests, events]),
      /private-token-123/,
    );
  });

  it("returns a safe result for an unknown tool", async () => {
    const model = scripted([
      call("unknown-1", "missing", {}),
      new AIMessage("That tool is unavailable."),
    ]);
    assert.equal(
      await runAgent("Use missing", model),
      "That tool is unavailable.",
    );
    assertToolResult(model, 1, "missing", "unknown-1", genericFailure);
  });

  it("recovers from malformed JSON separated into invalid_tool_calls by LangChain", async () => {
    const model = scripted([
      new AIMessage({
        content: "",
        invalid_tool_calls: [
          { id: "bad-json", name: "calculate", args: '{"operation":' },
        ],
      }),
      call("fixed", "calculate", { operation: "add", left: 1, right: 2 }),
      new AIMessage("3"),
    ]);
    assert.equal(await runAgent("Add one and two", model), "3");
    assertToolResult(model, 1, "calculate", "bad-json", genericFailure);
    assertToolResult(model, 2, "calculate", "fixed", '{"result":3}');
  });

  it("limits model calls precisely and executes the last allowed call's tools", async () => {
    const events: ToolCallEvent[] = [];
    const model = new ScriptedModel(() =>
      call(`call-${model.requests.length}`, "calculate", {
        operation: "add",
        left: 1,
        right: 1,
      }),
    );
    await assert.rejects(
      () =>
        runAgent("Keep calculating", model, {
          maxSteps: 2,
          onToolCall: (event) => events.push(event),
        }),
      /2-step limit/,
    );
    assert.equal(model.requests.length, 2);
    assert.equal(events.length, 2);
  });

  it("executes repeated call IDs instead of silently reusing old results", async () => {
    const model = scripted([
      call("repeated1", "calculate", { operation: "add", left: 1, right: 1 }),
      new AIMessage("2"),
      call("repeated1", "calculate", { operation: "add", left: 2, right: 3 }),
      new AIMessage("5"),
    ]);
    const session = new AgentSession(model);
    assert.equal(await session.send("1+1"), "2");
    assert.equal(await session.send("2+3"), "5");
    const result = model.requests[3]?.at(-1);
    assert.ok(ToolMessage.isInstance(result));
    assert.equal(result.content, '{"result":5}');
    assert.notEqual(result.tool_call_id, "repeated1");
    assert.match(result.tool_call_id, /^[a-z0-9]{9}$/);
  });

  it("aborts observer failures without exposing them to the model", async () => {
    const model = scripted([
      call("observer1", "calculate", { operation: "add", left: 1, right: 1 }),
      new AIMessage("Recovered."),
    ]);
    const session = new AgentSession(model, {
      onToolCall: () => {
        throw new Error("observer private-token-123");
      },
    });
    await assert.rejects(() => session.send("Failed"), /Tool execution failed/);
    assert.equal(model.requests.length, 1);
    assert.equal(await session.send("Fresh"), "Recovered.");
    assert.deepEqual(
      model.requests[1]?.slice(1).map((message) => message.content),
      ["Fresh"],
    );
    assert.doesNotMatch(JSON.stringify(model.requests), /private-token-123/);
  });

  it("accepts a final answer on the last allowed model call", async () => {
    const model = scripted([
      call("last", "calculate", { operation: "add", left: 1, right: 1 }),
      new AIMessage("2"),
    ]);
    assert.equal(await runAgent("1+1", model, 2), "2");
  });

  it("serializes multiple requested tools in their original order", async () => {
    const order: string[] = [];
    const ordered = defineTool({
      name: "ordered",
      description: "Track execution.",
      argumentsSchema: z.object({ value: z.string() }),
      execute: async ({ value }) => {
        order.push(`start:${value}`);
        await new Promise((resolve) => setTimeout(resolve, 5));
        order.push(`end:${value}`);
        return value;
      },
    });
    const model = scripted([
      new AIMessage({
        content: "",
        tool_calls: [
          { id: "one", name: "ordered", args: { value: "1" } },
          { id: "two", name: "ordered", args: { value: "2" } },
        ],
      }),
      new AIMessage("Done."),
    ]);
    await runAgent("Run both", model, {
      toolRegistry: createToolRegistry([ordered]),
    });
    assert.deepEqual(order, ["start:1", "end:1", "start:2", "end:2"]);
  });

  it("rejects calls without IDs before any tool runs", async () => {
    const events: ToolCallEvent[] = [];
    const model = scripted([
      new AIMessage({
        content: "",
        tool_calls: [
          { name: "calculate", args: { operation: "add", left: 1, right: 1 } },
        ],
      }),
    ]);
    await assert.rejects(() =>
      runAgent("1+1", model, { onToolCall: (event) => events.push(event) }),
    );
    assert.deepEqual(events, []);
  });
});

describe("AgentSession", () => {
  it("includes completed turns and their tool results in subsequent requests", async () => {
    const model = scripted([
      call("remembered", "calculate", { operation: "add", left: 1, right: 2 }),
      new AIMessage("The result is 3."),
      new AIMessage("I said 3."),
    ]);
    const session = new AgentSession(model);
    await session.send("1+2");
    assert.equal(await session.send("What did you just say?"), "I said 3.");
    const messages = model.requests[2];
    assert.ok(messages);
    assert.deepEqual(
      messages.map((message) => message.type),
      ["system", "human", "ai", "tool", "ai", "human"],
    );
    assert.equal(messages[3]?.content, '{"result":3}');
    assert.equal(messages[4]?.content, "The result is 3.");
  });

  it("discards every message from a failed turn while retaining completed turns", async () => {
    let completion = 0;
    const model = new ScriptedModel(() => {
      completion += 1;
      if (completion === 1) return new AIMessage("Remember this.");
      if (completion === 2)
        return call("failed-call", "calculate", {
          operation: "add",
          left: 1,
          right: 1,
        });
      if (completion === 3) throw new Error("model unavailable");
      return new AIMessage("Recovered.");
    });
    const session = new AgentSession(model);
    await session.send("Successful prompt");
    await assert.rejects(() => session.send("Failed prompt"), /unavailable/);
    assert.equal(await session.send("Fresh prompt"), "Recovered.");
    assert.deepEqual(
      model.requests[3]?.slice(1).map((message) => message.content),
      ["Successful prompt", "Remember this.", "Fresh prompt"],
    );
  });

  it("resets the model budget after a failed turn", async () => {
    const model = scripted([
      call("too-long", "calculate", { operation: "add", left: 1, right: 1 }),
      new AIMessage("Recovered."),
    ]);
    const session = new AgentSession(model, { maxSteps: 1 });
    await assert.rejects(() => session.send("Failed prompt"), /1-step limit/);
    assert.equal(await session.send("Fresh prompt"), "Recovered.");
    assert.deepEqual(
      model.requests[1]?.slice(1).map((message) => message.content),
      ["Fresh prompt"],
    );
  });

  it("rejects empty answers and discards the failed turn", async () => {
    const model = scripted([new AIMessage(" "), new AIMessage("Recovered.")]);
    const session = new AgentSession(model);
    await assert.rejects(
      () => session.send("Failed prompt"),
      /no text content/,
    );
    await session.send("Fresh prompt");
    assert.deepEqual(
      model.requests[1]?.slice(1).map((message) => message.content),
      ["Fresh prompt"],
    );
  });

  it("rejects concurrent turns without corrupting history", async () => {
    let release!: (message: AIMessage) => void;
    const model = new ScriptedModel(
      () =>
        new Promise<AIMessage>((resolve) => {
          release = resolve;
        }),
    );
    const session = new AgentSession(model);
    const first = session.send("First");
    await assert.rejects(() => session.send("Second"), /already in progress/);
    while (!release) await new Promise((resolve) => setImmediate(resolve));
    release(new AIMessage("Done."));
    assert.equal(await first, "Done.");
    assert.equal(model.requests.length, 1);
  });
});
