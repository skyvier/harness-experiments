# AI harness experiment

A minimal TypeScript agent that can call a validated calculator tool before
returning its answer. Inference is pinned to Mistral's EU endpoint.

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
```

## Verify

```sh
npm test
npm run check
```

The prototype is intentionally stateless between CLI invocations. Within one
invocation, it runs a bounded five-step agent loop and exposes one allowlisted
calculator tool.
