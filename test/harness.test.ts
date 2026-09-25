import assert from "node:assert/strict";
import { describe, it } from "node:test";

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
