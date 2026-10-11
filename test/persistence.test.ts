import assert from "node:assert/strict";
import { mkdtemp } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, it } from "node:test";

import { AIMessage, HumanMessage, ToolMessage } from "@langchain/core/messages";

import { AgentSession, validateConversationMessages } from "../src/agent.js";
import { openSqliteAgentPersistence } from "../src/persistence.js";
import { ScriptedModel } from "./helpers/model.js";

function scripted(responses: (AIMessage | Error)[]): ScriptedModel {
  return new ScriptedModel(() => {
    const response = responses.shift();
    assert.ok(response);
    if (response instanceof Error) throw response;
    return response;
  });
}

async function databasePath(): Promise<string> {
  return join(
    await mkdtemp(join(tmpdir(), "harness-checkpoints-")),
    "chat.sqlite",
  );
}

describe("durable LangGraph threads", () => {
  it("restores completed messages and tool results after a restart", async () => {
    const path = await databasePath();
    const firstPersistence = await openSqliteAgentPersistence(
      path,
      "project-a",
    );
    const firstModel = scripted([
      new AIMessage({
        content: "",
        tool_calls: [
          {
            id: "sum-call",
            name: "calculate",
            args: { operation: "add", left: 1, right: 2 },
          },
        ],
      }),
      new AIMessage("The result is 3."),
    ]);
    assert.equal(
      await new AgentSession(firstModel, {
        persistence: firstPersistence,
      }).send("What is 1 + 2?"),
      "The result is 3.",
    );
    firstPersistence.close();

    const secondPersistence = await openSqliteAgentPersistence(
      path,
      "project-a",
    );
    const secondModel = scripted([new AIMessage("I previously calculated 3.")]);
    assert.equal(
      await new AgentSession(secondModel, {
        persistence: secondPersistence,
      }).send("What did you calculate?"),
      "I previously calculated 3.",
    );

    const restored = secondModel.requests[0];
    assert.ok(restored);
    assert.deepEqual(
      restored.slice(1).map((message) => message.getType()),
      ["human", "ai", "tool", "ai", "human"],
    );
    assert.equal(restored[3]?.content, '{"result":3}');
    secondPersistence.close();
  });

  it("isolates histories by thread identifier", async () => {
    const path = await databasePath();
    const firstPersistence = await openSqliteAgentPersistence(path, "thread-a");
    await new AgentSession(scripted([new AIMessage("Thread A answer")]), {
      persistence: firstPersistence,
    }).send("Thread A prompt");
    firstPersistence.close();

    const secondPersistence = await openSqliteAgentPersistence(
      path,
      "thread-b",
    );
    const secondModel = scripted([new AIMessage("Thread B answer")]);
    await new AgentSession(secondModel, {
      persistence: secondPersistence,
    }).send("Thread B prompt");
    assert.deepEqual(
      secondModel.requests[0]?.slice(1).map((message) => message.content),
      ["Thread B prompt"],
    );
    secondPersistence.close();
  });

  it("rolls a failed turn back before the process restarts", async () => {
    const path = await databasePath();
    const failedPersistence = await openSqliteAgentPersistence(
      path,
      "rollback",
    );
    const failedModel = scripted([
      new AIMessage({
        content: "",
        tool_calls: [
          {
            id: "failed-call",
            name: "calculate",
            args: { operation: "add", left: 1, right: 1 },
          },
        ],
      }),
      new Error("model unavailable"),
    ]);
    await assert.rejects(
      () =>
        new AgentSession(failedModel, {
          persistence: failedPersistence,
        }).send("Failed prompt"),
      /unavailable/,
    );
    assert.equal(
      await failedPersistence.checkpointer.getTuple({
        configurable: { thread_id: "rollback" },
      }),
      undefined,
    );
    failedPersistence.close();

    const recoveredPersistence = await openSqliteAgentPersistence(
      path,
      "rollback",
    );
    const recoveredModel = scripted([new AIMessage("Recovered.")]);
    assert.equal(
      await new AgentSession(recoveredModel, {
        persistence: recoveredPersistence,
      }).send("Fresh prompt"),
      "Recovered.",
    );
    assert.deepEqual(
      recoveredModel.requests[0]?.slice(1).map((message) => message.content),
      ["Fresh prompt"],
    );
    recoveredPersistence.close();
  });

  it("rejects malformed restored messages before model execution", () => {
    assert.throws(
      () =>
        validateConversationMessages([
          new HumanMessage({ content: [{ text: "missing type" }] as never }),
        ]),
      /Invalid persisted message content/,
    );
    assert.deepEqual(
      validateConversationMessages([
        new HumanMessage("hello"),
        new AIMessage("hi"),
        new ToolMessage({
          name: "calculate",
          tool_call_id: "call-1",
          content: '{"result":2}',
        }),
      ]).map((message) => message.getType()),
      ["human", "ai", "tool"],
    );
  });
});
