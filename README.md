# AI harness experiment

A minimal TypeScript agent that can call validated calculator and weather tools
before returning its answer. Inference is pinned to Mistral's EU endpoint.

## Development environment

Enter the reproducible Nix development shell:

```sh
nix develop
```

The shell provides Node.js 24 and npm. Without Nix, Node.js 22 or newer is
required.

## Setup

```sh
npm install
cp .env.example .env
```

Put your API key in `.env`:

```dotenv
MISTRAL_API_KEY=your-api-key
MISTRAL_MODEL=ministral-8b-latest
```

The `.env` file is ignored by Git. `MISTRAL_MODEL` is optional and defaults to
the inexpensive `ministral-8b-latest` model.

## Run

```sh
npm start -- "Explain what an AI harness is in two sentences."
npm start -- "What is the weather at latitude 60.17 and longitude 24.94?"
```

Start an in-memory interactive conversation with:

```sh
npm start -- --chat
```

Enter `/exit` to end the conversation. Completed turns remain in context for
the rest of the process; failed turns are discarded from conversation history.

Resume a named conversation across process restarts with:

```sh
npm start -- --chat --thread my-project
```

Thread IDs may contain letters, numbers, dots, underscores, and hyphens. Durable
threads are stored in `.harness/conversations.sqlite`, which is ignored by Git.
Set `HARNESS_DB_PATH` to use another location. Different thread IDs have isolated
histories, and a failed turn is rolled back before the process accepts another
turn or exits.

Executed tool calls are logged to stderr, while the final answer is written to
stdout.

## Verify

```sh
npm test
npm run check
```

One-shot prompts and plain `--chat` sessions remain stateless between CLI
invocations; named threads opt into local persistence. Each user turn runs a
bounded five-step agent loop and exposes
allowlisted calculator and current-weather tools. Weather observations come
from the [Finnish Meteorological Institute's open data](https://en.ilmatieteenlaitos.fi/open-data)
under its stated Creative Commons licence. FMI responses are validated before
they enter application logic, and only normalized numeric observations are
sent to the model. Validation and execution failures are returned to the model
as tool results, allowing it to explain the failure or retry with a corrected
call. Tools may map recognized failures to safe descriptions; all other
exceptions are replaced with a generic error before entering the model context.

## LangChain architecture

The harness now uses TypeScript's `createAgent` from `langchain`, backed by
LangGraph, and `ChatMistralAI` from `@langchain/mistralai`. The terminal commands,
calculator, FMI adapter, and model-safe tool failure format stay the same.

- `src/agent.ts` wraps the framework in `AgentSession`. In-memory sessions pass
  completed history into each graph invocation. Durable sessions let LangGraph
  restore the named thread from SQLite. In both modes, history is committed only
  after a nonempty final answer, so failed turns never enter later context.
- The `HarnessBoundaries` middleware counts model calls (five per turn by
  default), validates model output with Zod, and dispatches tool calls through
  the existing validated registry. It deliberately handles unknown tools and
  malformed JSON arguments with the same safe failure results as before.
  Tool execution remains sequential, even if a model requests multiple calls.
- `src/tool.ts` advertises provider-independent JSON Schema function definitions
  to LangChain. Zod remains the authority for validating executor arguments;
  the calculator and weather code do not depend on LangChain.
- `src/harness.ts` explicitly configures `https://api.eu.mistral.ai`. A Mistral
  request hook preserves `parallel_tool_calls: false`, since the LangChain
  adapter does not currently expose that setting. Provider responses can stream
  internally; the terminal still prints only the completed answer.
- `src/config.ts` validates environment settings with Zod. The default model
  remains `ministral-8b-latest`.

`runAgent` still supports isolated turns and the numeric step-limit shorthand,
while its injectable model boundary is now LangChain's `BaseChatModel`.
Tests exercise the real framework graph with a scripted model and verify the
Mistral HTTP boundary with mocked streaming responses; no API key or live
inference is needed for the test suite.

Named sessions use LangGraph's SQLite checkpointer. Each turn runs inside a
database transaction so model failures, tool failures, and invalid final answers
cannot leave partial checkpoints. Restored message state is validated with Zod
before it reaches the model. The project still has no external tracing service
or long-term memory shared across threads.
