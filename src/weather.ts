import { z } from "zod";

import { defineTool } from "./tool.js";

const FMI_TIMESERIES_URL = "https://opendata.fmi.fi/timeseries";
const OBSERVATION_WINDOW_MS = 3 * 60 * 60 * 1_000;
const REQUEST_TIMEOUT_MS = 5_000;
const MAX_RESPONSE_BYTES = 64 * 1_024;

const weatherArgumentsSchema = z
  .object({
    latitude: z.number().finite().min(-90).max(90),
    longitude: z.number().finite().min(-180).max(180),
  })
  .strict();

const fmiTimestampSchema = z
  .string()
  .regex(/^\d{8}T\d{6}$/)
  .refine(isValidFmiTimestamp, "Invalid FMI timestamp");

const fmiResponseSchema = z
  .array(
    z
      .object({
        utctime: fmiTimestampSchema,
        t2m: z.number().finite().min(-100).max(70).nullable(),
        ws_10min: z.number().finite().min(0).max(150).nullable(),
      })
      .strict(),
  )
  .max(1_000);

/** Geographic coordinates accepted by the weather capability. */
export type WeatherCoordinates = Readonly<
  z.infer<typeof weatherArgumentsSchema>
>;

/** A current weather observation in provider-independent units. */
export interface WeatherObservation {
  readonly observedAt: string;
  readonly temperatureC: number;
  readonly windSpeedMps: number;
}

/** Application-owned boundary for retrieving current weather. */
export interface WeatherClient {
  getCurrentWeather(
    coordinates: WeatherCoordinates,
  ): Promise<WeatherObservation>;
}

type Fetch = (input: string | URL, init?: RequestInit) => Promise<Response>;

interface FmiWeatherClientOptions {
  readonly fetch?: Fetch;
  readonly now?: () => Date;
}

/** Creates an FMI Open Data adapter for the application weather contract. */
export function createFmiWeatherClient(
  options: FmiWeatherClientOptions = {},
): WeatherClient {
  const fetch_ = options.fetch ?? globalThis.fetch;
  const now = options.now ?? (() => new Date());

  return {
    async getCurrentWeather(
      coordinates: WeatherCoordinates,
    ): Promise<WeatherObservation> {
      try {
        const requestTime = now();
        const response = await fetch_(createFmiUrl(coordinates, requestTime), {
          headers: { accept: "application/json" },
          signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
        });

        if (!response.ok) {
          throw new FmiWeatherError(`FMI returned HTTP ${response.status}.`);
        }

        const payload = await readJson(response);
        const observations = fmiResponseSchema.parse(payload);
        const latest = observations
          .filter(
            (observation): observation is typeof observation & {
              t2m: number;
              ws_10min: number;
            } => observation.t2m !== null && observation.ws_10min !== null,
          )
          .sort((left, right) => right.utctime.localeCompare(left.utctime))[0];

        if (!latest) {
          throw new FmiWeatherError("FMI returned no complete observations.");
        }

        return {
          observedAt: toIsoTimestamp(latest.utctime),
          temperatureC: latest.t2m,
          windSpeedMps: latest.ws_10min,
        };
      } catch (error: unknown) {
        if (error instanceof FmiWeatherError) {
          throw error;
        }
        throw new FmiWeatherError("FMI weather lookup failed.", { cause: error });
      }
    },
  };
}

/** Creates the model-facing weather tool around a substitutable weather client. */
export function createWeatherAgentTool(client: WeatherClient) {
  return defineTool({
    name: "get_weather",
    description:
      "Get the latest FMI weather observation near coordinates in Finland.",
    argumentsSchema: weatherArgumentsSchema,
    async execute(coordinates: WeatherCoordinates): Promise<string> {
      return JSON.stringify(await client.getCurrentWeather(coordinates));
    },
    mapError: (error) =>
      error instanceof FmiWeatherError
        ? {
            code: "weather_unavailable",
            message: "Current weather observations are temporarily unavailable.",
            retryable: true,
          }
        : undefined,
  });
}

export const weatherAgentTool = createWeatherAgentTool(
  createFmiWeatherClient(),
);

function createFmiUrl(
  { latitude, longitude }: WeatherCoordinates,
  requestTime: Date,
): URL {
  const url = new URL(FMI_TIMESERIES_URL);
  url.search = new URLSearchParams({
    producer: "opendata",
    latlon: `${latitude},${longitude}`,
    param: "utctime,t2m,ws_10min",
    format: "json",
    precision: "double",
    tz: "UTC",
    timestep: "data",
    starttime: new Date(
      requestTime.getTime() - OBSERVATION_WINDOW_MS,
    ).toISOString(),
    endtime: requestTime.toISOString(),
  }).toString();
  return url;
}

async function readJson(response: Response): Promise<unknown> {
  const declaredLength = Number(response.headers.get("content-length"));
  if (Number.isFinite(declaredLength) && declaredLength > MAX_RESPONSE_BYTES) {
    throw new FmiWeatherError("FMI response is too large.");
  }

  const body = await response.text();
  if (Buffer.byteLength(body, "utf8") > MAX_RESPONSE_BYTES) {
    throw new FmiWeatherError("FMI response is too large.");
  }

  return JSON.parse(body) as unknown;
}

function isValidFmiTimestamp(value: string): boolean {
  const iso = toIsoTimestamp(value);
  return (
    !Number.isNaN(Date.parse(iso)) && toFmiTimestamp(new Date(iso)) === value
  );
}

function toIsoTimestamp(value: string): string {
  return `${value.slice(0, 4)}-${value.slice(4, 6)}-${value.slice(6, 8)}T${value.slice(9, 11)}:${value.slice(11, 13)}:${value.slice(13, 15)}Z`;
}

function toFmiTimestamp(value: Date): string {
  return value
    .toISOString()
    .replace(/[-:]/g, "")
    .replace(".000Z", "");
}

class FmiWeatherError extends Error {}
