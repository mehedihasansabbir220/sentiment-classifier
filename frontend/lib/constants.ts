/**
 * Base URL of the FastAPI backend.
 *
 * Read once here so no component ever hardcodes it. `NEXT_PUBLIC_` variables
 * are inlined at build time, so the reference must stay a literal
 * `process.env.NEXT_PUBLIC_API_URL` expression.
 */
export const API_BASE_URL = (
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"
).replace(/\/+$/, "");

/** Matches MAX_TEXT_LENGTH in the backend schema. */
export const MAX_TEXT_LENGTH = 10_000;

/** Give up on a prediction after this long. */
export const REQUEST_TIMEOUT_MS = 15_000;

/** Below this the counter turns amber to warn the limit is close. */
export const COUNTER_WARNING_RATIO = 0.9;

export const formatPercent = (value: number): string =>
  `${(value * 100).toFixed(1)}%`;
