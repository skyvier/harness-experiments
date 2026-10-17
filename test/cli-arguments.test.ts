import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { parseCliArguments } from "../src/cli-arguments.js";

describe("CLI arguments", () => {
  it("selects one-shot and ephemeral chat modes", () => {
    assert.deepEqual(parseCliArguments(["hello", "world"]), {
      mode: "prompt",
      prompt: "hello world",
    });
    assert.deepEqual(parseCliArguments(["--chat"]), { mode: "chat" });
  });

  it("validates a durable thread identifier", () => {
    assert.deepEqual(
      parseCliArguments(["--chat", "--thread", "my.project_1"]),
      { mode: "durable-chat", threadId: "my.project_1" },
    );
    assert.throws(
      () => parseCliArguments(["--chat", "--thread", "bad/thread"]),
      /Thread IDs/,
    );
  });

  it("rejects incomplete or misplaced thread options", () => {
    assert.throws(() => parseCliArguments([]), /Usage/);
    assert.throws(() => parseCliArguments(["--chat", "--thread"]), /Usage/);
    assert.throws(() => parseCliArguments(["hello", "--thread"]), /Usage/);
  });
});
