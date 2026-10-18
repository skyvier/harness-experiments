import { createInterface } from "node:readline";

/** Minimal conversation capability required by the terminal chat loop. */
export interface ChatSession {
  send(prompt: string): Promise<string>;
}

/** Runs an in-memory chat until end-of-input or the user enters `/exit`. */
export async function runInteractiveChat(
  session: ChatSession,
  input: NodeJS.ReadableStream,
  output: NodeJS.WritableStream,
  errorOutput: NodeJS.WritableStream,
): Promise<void> {
  const lines = createInterface({ input, crlfDelay: Infinity });
  output.write("Interactive chat started. Type /exit to quit.\n> ");

  try {
    for await (const line of lines) {
      const prompt = line.trim();
      if (prompt === "/exit") {
        return;
      }

      if (prompt) {
        try {
          output.write(`${await session.send(prompt)}\n`);
        } catch (error: unknown) {
          const message = error instanceof Error ? error.message : String(error);
          errorOutput.write(`Error: ${message}\n`);
        }
      }

      output.write("> ");
    }
  } finally {
    lines.close();
  }
}
