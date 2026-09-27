import assert from "node:assert/strict";
import { Readable, Writable } from "node:stream";
import { describe, it } from "node:test";

import { runInteractiveChat } from "../src/interactive-chat.js";

describe("interactive chat", () => {
  it("continues after a failed turn and exits on command", async () => {
    const prompts: string[] = [];
    const output = captureOutput();
    const errors = captureOutput();

    await runInteractiveChat(
      {
        async send(prompt) {
          prompts.push(prompt);
          if (prompt === "fail") {
            throw new Error("temporary failure");
          }
          return `reply to ${prompt}`;
        },
      },
      Readable.from(["first\nfail\nsecond\n/exit\nignored\n"]),
      output.stream,
      errors.stream,
    );

    assert.deepEqual(prompts, ["first", "fail", "second"]);
    assert.match(output.text(), /reply to first/);
    assert.match(output.text(), /reply to second/);
    assert.doesNotMatch(output.text(), /ignored/);
    assert.match(errors.text(), /Error: temporary failure/);
  });
});

function captureOutput(): {
  readonly stream: Writable;
  readonly text: () => string;
} {
  let contents = "";
  return {
    stream: new Writable({
      write(chunk, _encoding, callback) {
        contents += String(chunk);
        callback();
      },
    }),
    text: () => contents,
  };
}
