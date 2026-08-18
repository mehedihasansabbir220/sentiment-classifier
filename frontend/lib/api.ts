import {
  API_BASE_URL,
  MAX_TEXT_LENGTH,
  REQUEST_TIMEOUT_MS,
} from "@/lib/constants";
import type {
  PredictionRequest,
  PredictionResponse,
  Probabilities,
  Sentiment,
} from "@/lib/types";

/** Text rejected before a request is made. */
export class ValidationError extends Error {}

/** The request was made but did not produce a usable prediction. */
export class ApiError extends Error {
  readonly status?: number;

  constructor(message: string, status?: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

/** Step 1 — validate before spending a round trip. */
export function validateText(text: string): string {
  const trimmed = text.trim();

  if (!trimmed) {
    throw new ValidationError("Enter some text before running the analysis.");
  }
  if (text.length > MAX_TEXT_LENGTH) {
    throw new ValidationError(
      `Text is too long. Keep it under ${MAX_TEXT_LENGTH.toLocaleString()} characters.`,
    );
  }
  return text;
}

/** FastAPI returns `detail` as a string, or as a list of validation errors. */
function readErrorDetail(body: unknown): string | null {
  if (typeof body !== "object" || body === null || !("detail" in body)) {
    return null;
  }

  const { detail } = body as { detail: unknown };

  if (typeof detail === "string") {
    return detail;
  }
  if (Array.isArray(detail)) {
    const first = detail.find(
      (entry): entry is { msg: string } =>
        typeof entry === "object" && entry !== null && typeof (entry as { msg?: unknown }).msg === "string",
    );
    return first ? first.msg.replace(/^Value error,\s*/, "") : null;
  }
  return null;
}

function messageForStatus(status: number, detail: string | null): string {
  switch (status) {
    case 422:
      return detail ?? "That text could not be analyzed. Try rewording it.";
    case 503:
      return "The sentiment model is not available. Check that the backend loaded the checkpoint.";
    case 500:
      return "The model failed to analyze this text. Please try again.";
    default:
      return detail ?? `The API responded with an unexpected status (${status}).`;
  }
}

/** Guards against a response that parses as JSON but is not a prediction. */
function assertPrediction(body: unknown): asserts body is PredictionResponse {
  const candidate = body as Partial<PredictionResponse> | null;
  const probabilities = candidate?.probabilities as Partial<Probabilities> | undefined;

  const isValid =
    !!candidate &&
    typeof candidate.text === "string" &&
    (candidate.sentiment === "positive" || candidate.sentiment === "negative") &&
    typeof candidate.confidence === "number" &&
    typeof probabilities?.positive === "number" &&
    typeof probabilities?.negative === "number";

  if (!isValid) {
    throw new ApiError("The API returned an unexpected response shape.");
  }
}

/**
 * Steps 3-7 — call `POST /predict` and return the prediction.
 *
 * Throws {@link ValidationError} for text the backend would reject anyway, and
 * {@link ApiError} for transport, status, or payload failures.
 */
export async function predictSentiment(text: string): Promise<PredictionResponse> {
  const payload: PredictionRequest = { text: validateText(text) };

  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
    });
  } catch (caught) {
    if (caught instanceof DOMException && caught.name === "TimeoutError") {
      throw new ApiError("The request timed out. The model may still be starting up.");
    }
    throw new ApiError(
      `Could not reach the API at ${API_BASE_URL}. Check that the backend is running.`,
    );
  }

  const body: unknown = await response.json().catch(() => null);

  if (!response.ok) {
    throw new ApiError(messageForStatus(response.status, readErrorDetail(body)), response.status);
  }

  assertPrediction(body);
  return body;
}

export type { PredictionResponse, Probabilities, Sentiment };
