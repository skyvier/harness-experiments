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

Executed tool calls are logged to stderr, while the final answer is written to
stdout.

## Verify

```sh
npm test
npm run check
```

The prototype is intentionally stateless between CLI invocations. Within one
invocation, it runs a bounded five-step agent loop and exposes allowlisted
calculator and current-weather tools. Weather observations come from the
[Finnish Meteorological Institute's open data](https://en.ilmatieteenlaitos.fi/open-data)
under its stated Creative Commons licence. FMI responses are validated before
they enter application logic, and only normalized numeric observations are sent
to the model. Validation and execution failures are returned to the model as
tool results, allowing it to explain the failure or retry with a corrected call.
Tools may map recognized failures to safe descriptions; all other exceptions
are replaced with a generic error before entering the model context.
