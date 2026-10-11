import { z } from "zod";

/** Runtime settings required to call the Mistral text-generation API. */
export interface HarnessConfig {
  readonly apiKey: string;
  readonly model: string;
}

/** Runtime settings used only by durable interactive chat. */
export interface PersistenceConfig {
  readonly databasePath: string;
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
const persistenceConfigSchema = z
  .object({
    HARNESS_DB_PATH: z.string().trim().min(1).max(4_096).optional(),
  })
  .transform((settings) => ({
    databasePath: settings.HARNESS_DB_PATH || ".harness/conversations.sqlite",
  }));

/** Validates untrusted environment variables before they reach the API client. */
export function readConfig(
  environment: NodeJS.ProcessEnv = process.env,
): HarnessConfig {
  return configSchema.parse(environment);
}

/** Validates durable-chat settings before opening local storage. */
export function readPersistenceConfig(
  environment: NodeJS.ProcessEnv = process.env,
): PersistenceConfig {
  return persistenceConfigSchema.parse(environment);
}
