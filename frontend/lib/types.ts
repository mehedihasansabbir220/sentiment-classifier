/** Shared contract with the FastAPI backend (`POST /predict`). */

export type Sentiment = "positive" | "negative";

/** Probability per class; the two values sum to 1. */
export interface Probabilities {
  positive: number;
  negative: number;
}

/** Request body for `POST /predict`. */
export interface PredictionRequest {
  text: string;
}

/** Response body from `POST /predict`. */
export interface PredictionResponse {
  text: string;
  sentiment: Sentiment;
  confidence: number;
  probabilities: Probabilities;
}
