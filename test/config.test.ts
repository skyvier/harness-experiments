import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { readConfig } from "../src/config.js";

describe("readConfig", () => {
  it("uses the cheap Ministral model by default", () => {
    assert.deepEqual(readConfig({ MISTRAL_API_KEY: "secret" }), {
      apiKey: "secret",
      model: "ministral-8b-latest",
    });
  });

  it("rejects a missing API key before making a request", () => {
    assert.throws(() => readConfig({}), /MISTRAL_API_KEY is required/);
  });

  it("accepts an explicit model override", () => {
    assert.equal(
      readConfig({
        MISTRAL_API_KEY: "secret",
        MISTRAL_MODEL: "mistral-small-latest",
      }).model,
      "mistral-small-latest",
    );
  });
});
