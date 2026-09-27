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

    assert.equal(await registry.execute("echo", '{"text":"hello"}'), "hello");
    await assert.rejects(() => registry.execute("echo", { text: 42 }));
  });

  it("rejects unknown and duplicate tool names", async () => {
    const noop = defineTool({
      name: "noop",
      description: "Do nothing.",
      argumentsSchema: z.object({}).strict(),
      execute: () => "done",
    });

    assert.throws(() => createToolRegistry([noop, noop]), /Duplicate tool/);
    await assert.rejects(
      () => createToolRegistry([noop]).execute("missing", {}),
      /Unknown tool/,
    );
  });
});
