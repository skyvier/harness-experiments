import assert from "node:assert/strict";
import { mkdtemp, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";
import type { SpawnSyncReturns } from "node:child_process";
import { describe, it } from "node:test";

import { z } from "zod";

const projectDirectory = fileURLToPath(new URL("..", import.meta.url));
const capturedRequestSchema = z.object({
  messages: z.array(
    z.object({
      role: z.string(),
      content: z.unknown().optional(),
    }),
  ),
});

describe("durable chat CLI", () => {
  it("restores a named thread in a new process", async () => {
    const directory = await mkdtemp(join(tmpdir(), "harness-cli-"));
    const databasePath = join(directory, "conversations.sqlite");
    const capturePath = join(directory, "requests.jsonl");
    const preloadPath = join(directory, "mock-mistral.mjs");
    await writeFile(preloadPath, mockMistralPreload, "utf8");

    const first = runChat(
      preloadPath,
      databasePath,
      capturePath,
      "Remember alpha\n/exit\n",
    );
    assert.equal(first.status, 0, first.stderr);
    assert.match(first.stdout, /Stored alpha\./);

    const second = runChat(
      preloadPath,
      databasePath,
      capturePath,
      "What did I say\?\n/exit\n",
    );
    assert.equal(second.status, 0, second.stderr);
    assert.match(second.stdout, /You said alpha\./);

    const requests = (await readFile(capturePath, "utf8"))
      .trim()
      .split("\n")
      .map((line) => capturedRequestSchema.parse(JSON.parse(line)));
    assert.equal(requests.length, 2);
    assert.deepEqual(
      requests[1]?.messages.slice(1).map((message) => message.content),
      ["Remember alpha", "Stored alpha.", "What did I say?"],
    );
  });
});

function runChat(
  preloadPath: string,
  databasePath: string,
  capturePath: string,
  input: string,
): SpawnSyncReturns<string> {
  return spawnSync(
    process.execPath,
    [
      "--import",
      "tsx",
      "--import",
      preloadPath,
      "src/cli.ts",
      "--chat",
      "--thread",
      "movie",
    ],
    {
      cwd: projectDirectory,
      encoding: "utf8",
      env: {
        ...process.env,
        HARNESS_CAPTURE_PATH: capturePath,
        HARNESS_DB_PATH: databasePath,
        MISTRAL_API_KEY: "test-key",
      },
      input,
    },
  );
}

const mockMistralPreload = String.raw`
import { appendFileSync } from "node:fs";

globalThis.fetch = async (input, init) => {
  const request = new Request(input, init);
  const body = JSON.parse(await request.clone().text());
  appendFileSync(process.env.HARNESS_CAPTURE_PATH, JSON.stringify(body) + "\n");
  const hasPriorPrompt = body.messages.some(
    (message) => message.content === "Remember alpha",
  );
  const content = hasPriorPrompt && body.messages.at(-1)?.content === "What did I say?"
    ? "You said alpha."
    : "Stored alpha.";
  const chunk = {
    id: "completion-test",
    object: "chat.completion.chunk",
    created: 1,
    model: "ministral-8b-latest",
    choices: [{
      index: 0,
      delta: { role: "assistant", content },
      finish_reason: "stop",
    }],
    usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
  };
  return new Response(
    "data: " + JSON.stringify(chunk) + "\n\ndata: [DONE]\n\n",
    { headers: { "content-type": "text/event-stream" } },
  );
};
`;
