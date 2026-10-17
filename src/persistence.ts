import { mkdir } from "node:fs/promises";
import { dirname } from "node:path";

import type { BaseCheckpointSaver } from "@langchain/langgraph-checkpoint";
import { SqliteSaver } from "@langchain/langgraph-checkpoint-sqlite";
import { z } from "zod";

const threadIdSchema = z
  .string()
  .trim()
  .min(1)
  .max(64)
  .regex(
    /^[A-Za-z0-9][A-Za-z0-9._-]*$/,
    "Thread IDs may contain letters, numbers, dots, underscores, and hyphens.",
  );

const databasePathSchema = z.string().trim().min(1).max(4_096);

/** Checkpointing controls consumed by one AgentSession. */
export interface AgentPersistence {
  readonly checkpointer: BaseCheckpointSaver;
  readonly threadId: string;
  transaction<Result>(work: () => Promise<Result>): Promise<Result>;
}

/** A lightweight SQLite checkpoint store for local persistent chat threads. */
export interface SqliteAgentPersistence extends AgentPersistence {
  close(): void;
}

/** Validates an untrusted CLI thread identifier. */
export function parseThreadId(value: unknown): string {
  return threadIdSchema.parse(value);
}

/** Validates an untrusted environment-provided checkpoint database path. */
export function parseDatabasePath(value: unknown): string {
  return databasePathSchema.parse(value);
}

/** Opens a LangGraph SQLite checkpointer with atomic turn transactions. */
export async function openSqliteAgentPersistence(
  databasePath: string,
  threadId: string,
): Promise<SqliteAgentPersistence> {
  const path = parseDatabasePath(databasePath);
  const validatedThreadId = parseThreadId(threadId);
  await mkdir(dirname(path), { recursive: true });

  const checkpointer = SqliteSaver.fromConnString(path);
  // Initialize tables and WAL mode before opening an application transaction.
  await checkpointer.getTuple({
    configurable: { thread_id: "__harness_setup__" },
  });

  let transactionTail: Promise<void> = Promise.resolve();

  return {
    checkpointer,
    threadId: validatedThreadId,
    async transaction<Result>(work: () => Promise<Result>): Promise<Result> {
      const previous = transactionTail;
      let release!: () => void;
      transactionTail = new Promise<void>((resolve) => {
        release = resolve;
      });
      await previous;

      try {
        checkpointer.db.exec("BEGIN IMMEDIATE");
        try {
          const result = await work();
          checkpointer.db.exec("COMMIT");
          return result;
        } catch (error: unknown) {
          if (checkpointer.db.inTransaction) checkpointer.db.exec("ROLLBACK");
          throw error;
        }
      } finally {
        release();
      }
    },
    close(): void {
      checkpointer.db.close();
    },
  };
}
