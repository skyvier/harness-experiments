import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { z } from "zod";
import { AgentSession } from "../src/agent.js";
import { createMistralModel } from "../src/harness.js";

import { extractText } from "../src/agent.js";

describe("extractText", () => {
  it("returns a plain-text response", () => {
    assert.equal(extractText("hello"), "hello");
  });

  it("joins text chunks and ignores non-text chunks", () => {
    assert.equal(
      extractText([
        { type: "thinking", thinking: [] },
        { type: "text", text: "hello " },
        { type: "text", text: "world" },
      ]),
      "hello world",
    );
  });

  it("rejects a response without text", () => {
    assert.throws(() => extractText(null), /no text content/);
  });
});

// Drive the actual provider integration through a mocked SDK transport.
// This checks wire behavior rather than just constructor properties.
describe("LangChain Mistral adapter", () => {
  it("pins EU inference and preserves provider settings across a tool round trip", async (context) => {
    const requests: Request[] = [];
    const bodies: Record<string, unknown>[] = [];
    const responses = [
      {
        role: "assistant",
        content: "",
        tool_calls: [
          {
            id: "calc12345",
            type: "function",
            function: {
              name: "calculate",
              arguments: '{"operation":"multiply","left":137,"right":42}',
            },
          },
        ],
      },
      { role: "assistant", content: "5754" },
    ];
    const model = createMistralModel({
      apiKey: "test-key",
      model: "ministral-8b-latest",
    });
    context.mock.method(
      globalThis,
      "fetch",
      async (input: RequestInfo | URL, init?: RequestInit) => {
        const request = new Request(input, init);
        requests.push(request);
        bodies.push(
          z.record(z.string(), z.unknown()).parse(await request.clone().json()),
        );
        const message = responses.shift();
        assert.ok(message);
        const chunk = {
          id: "completion-test",
          object: "chat.completion.chunk",
          created: 1,
          model: "ministral-8b-latest",
          choices: [
            {
              index: 0,
              delta: message,
              finish_reason: responses.length ? "tool_calls" : "stop",
            },
          ],
          usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
        };
        return new Response(
          `data: ${JSON.stringify(chunk)}\n\ndata: [DONE]\n\n`,
          {
            headers: { "content-type": "text/event-stream" },
          },
        );
      },
    );
    assert.equal(await new AgentSession(model).send("137 * 42"), "5754");
    assert.equal(requests.length, 2);
    for (const request of requests) {
      assert.equal(new URL(request.url).origin, "https://api.eu.mistral.ai");
      assert.equal(new URL(request.url).pathname, "/v1/chat/completions");
      assert.equal(request.headers.get("authorization"), "Bearer test-key");
    }
    for (const body of bodies) {
      assert.equal(body.model, "ministral-8b-latest");
      assert.equal(body.tool_choice, "auto");
      assert.equal(body.parallel_tool_calls, false);
      const tools = z
        .array(z.object({ function: z.object({ name: z.string() }) }))
        .parse(body.tools);
      assert.deepEqual(
        tools.map((tool) => tool.function.name),
        ["calculate", "get_weather"],
      );
    }
    const messages = z
      .array(
        z.object({
          role: z.string(),
          content: z.unknown().optional(),
          tool_call_id: z.string().optional(),
        }),
      )
      .parse(bodies[1]?.messages);
    assert.equal(messages.at(-1)?.role, "tool");
    assert.equal(messages.at(-1)?.tool_call_id, "calc12345");
    assert.equal(messages.at(-1)?.content, '{"result":5754}');
  });

  it("rejects malformed provider responses before tools execute", async (context) => {
    const model = createMistralModel({
      apiKey: "test-key",
      model: "ministral-8b-latest",
    });
    context.mock.method(globalThis, "fetch", async () =>
      Response.json({ choices: "invalid" }),
    );
    await assert.rejects(() => new AgentSession(model).send("Hello"));
  });
});
