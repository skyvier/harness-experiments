import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { z } from "zod";

import { createToolRegistry, defineTool } from "../src/tool.js";

describe("tool registry", () => {
  it("validates arguments before calling a typed executor", async () => {
    const echo = defineTool({
      name: "echo",
      description: "Echo text.",
      argumentsSchema: z.object({ text: z.string() }).strict(),
      execute: ({ text }) => text,
    });
    const registry = createToolRegistry([echo]);

    assert.deepEqual(await registry.execute("echo", '{"text":"hello"}'), {
      ok: true,
      result: "hello",
    });
    assert.deepEqual(await registry.execute("echo", { text: 42 }), {
      ok: false,
      failure: {
        code: "tool_error",
        message: "The tool encountered an error.",
        retryable: false,
      },
    });
  });

  it("hides unknown tools and rejects duplicate names", async () => {
    const noop = defineTool({
      name: "noop",
      description: "Do nothing.",
      argumentsSchema: z.object({}).strict(),
      execute: () => "done",
    });

    assert.throws(() => createToolRegistry([noop, noop]), /Duplicate tool/);
    assert.deepEqual(
      await createToolRegistry([noop]).execute("missing", {}),
      {
        ok: false,
        failure: {
          code: "tool_error",
          message: "The tool encountered an error.",
          retryable: false,
        },
      },
    );
  });

  it("uses a tool-owned mapping for recognized failures", async () => {
    const lookup = defineTool({
      name: "lookup",
      description: "Look up data.",
      argumentsSchema: z.object({}).strict(),
      execute: () => {
        throw new Error("upstream response contained secret-123");
      },
      mapError: (error) =>
        error instanceof Error
          ? {
              code: "service_unavailable",
              message: "The lookup service is temporarily unavailable.",
              retryable: true,
            }
          : undefined,
    });

    const result = await createToolRegistry([lookup]).execute("lookup", {});

    assert.deepEqual(result, {
      ok: false,
      failure: {
        code: "service_unavailable",
        message: "The lookup service is temporarily unavailable.",
        retryable: true,
      },
    });
    assert.doesNotMatch(JSON.stringify(result), /secret-123/);
  });

  it("uses the generic failure when a tool error mapper fails", async () => {
    const broken = defineTool({
      name: "broken",
      description: "Fail unsafely.",
      argumentsSchema: z.object({}).strict(),
      execute: () => {
        throw new Error("private failure details");
      },
      mapError: () => {
        throw new Error("broken mapper details");
      },
    });

    const result = await createToolRegistry([broken]).execute("broken", {});

    assert.deepEqual(result, {
      ok: false,
      failure: {
        code: "tool_error",
        message: "The tool encountered an error.",
        retryable: false,
      },
    });
    assert.doesNotMatch(JSON.stringify(result), /private|broken mapper/);
  });
});
