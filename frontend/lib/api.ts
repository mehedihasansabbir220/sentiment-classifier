import {
  API_BASE_URL,
  MAX_TEXT_LENGTH,
  REQUEST_TIMEOUT_MS,
} from "@/lib/constants";
import {
  ApiError,
  AppError,
  ValidationError,
  errorFromFetchFailure,
  errorFromHttp,
  errorFromInvalidResponse,
  errorFromUnknown,
} from "@/lib/errors";
import type { PredictionRequest, PredictionResponse } from "@/lib/types";

export { ApiError, AppError, ValidationError, errorFromUnknown };

/** Step 1 — validate before spending a round trip. */
export function validateText(text: string): string {
  const trimmed = text.trim();

  if (!trimmed) {
    throw new ValidationError("EMPTY_TEXT");
  }
  if (text.length > MAX_TEXT_LENGTH) {
    throw new ValidationError("TEXT_TOO_LONG");
  }
  return text;
}

/** Guards against a response that parses as JSON but is not a prediction. */
function assertPrediction(body: unknown): asserts body is PredictionResponse {
  const candidate = body as Partial<PredictionResponse> | null;
  const probabilities = candidate?.probabilities;

  const isValid =
    !!candidate &&
    typeof candidate.text === "string" &&
    (candidate.sentiment === "positive" || candidate.sentiment === "negative") &&
    typeof candidate.confidence === "number" &&
    typeof probabilities === "object" &&
    probabilities !== null &&
    typeof probabilities.positive === "number" &&
    typeof probabilities.negative === "number";

  if (!isValid) {
    throw errorFromInvalidResponse();
  }
}

/**
 * Call `POST /predict` and return the prediction.
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
    throw errorFromFetchFailure(caught);
  }

  const body: unknown = await response.json().catch(() => null);

  if (!response.ok) {
    throw errorFromHttp(response.status, body);
  }

  assertPrediction(body);
  return body;
}

export type { PredictionResponse } from "@/lib/types";
