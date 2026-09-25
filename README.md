# AI harness experiment

A minimal TypeScript CLI that sends one prompt to Mistral and prints the
response. Inference is pinned to Mistral's EU endpoint.

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

This first prototype is intentionally stateless: it makes one chat-completion
request, has no tools, and does not retain conversation history.
