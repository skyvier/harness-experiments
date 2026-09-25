/** Runtime settings required to call the Mistral text-generation API. */
export interface HarnessConfig {
  readonly apiKey: string;
  readonly model: string;
}

const DEFAULT_MODEL = "ministral-8b-latest";

/** Validates untrusted environment variables before they reach the API client. */
export function readConfig(
  environment: NodeJS.ProcessEnv = process.env,
): HarnessConfig {
  const apiKey = environment.MISTRAL_API_KEY?.trim();

  if (!apiKey) {
    throw new Error(
      "MISTRAL_API_KEY is required. Copy .env.example to .env and add your key.",
    );
  }

  const model = environment.MISTRAL_MODEL?.trim() || DEFAULT_MODEL;

  return { apiKey, model };
}
