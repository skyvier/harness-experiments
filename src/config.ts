import { z } from "zod";

/** Runtime settings required to call the Mistral text-generation API. */
export interface HarnessConfig {
  readonly apiKey: string;
  readonly model: string;
}

const DEFAULT_MODEL = "ministral-8b-latest";
const REQUIRED_KEY =
  "MISTRAL_API_KEY is required. Copy .env.example to .env and add your key.";
const configSchema = z
  .object({
    MISTRAL_API_KEY: z
      .string({ error: REQUIRED_KEY })
      .trim()
      .min(1, REQUIRED_KEY),
    MISTRAL_MODEL: z.string().trim().optional(),
  })
  .transform((settings) => ({
    apiKey: settings.MISTRAL_API_KEY,
    model: settings.MISTRAL_MODEL || DEFAULT_MODEL,
  }));

/** Validates untrusted environment variables before they reach the API client. */
export function readConfig(
  environment: NodeJS.ProcessEnv = process.env,
): HarnessConfig {
  return configSchema.parse(environment);
}
