import { MAX_TEXT_LENGTH } from "@/lib/constants";

export type ErrorCode =
  | "MODEL_NOT_FOUND"
  | "TOKENIZER_FAILURE"
  | "MODEL_LOAD_FAILURE"
  | "INVALID_REQUEST"
  | "EMPTY_TEXT"
  | "TEXT_TOO_LONG"
  | "INFERENCE_FAILURE"
  | "INTERNAL_ERROR"
  | "BACKEND_UNAVAILABLE"
  | "NETWORK_FAILURE"
  | "TIMEOUT"
  | "HTTP_ERROR"
  | "INVALID_RESPONSE";

const USER_MESSAGES: Record<ErrorCode, string> = {
  MODEL_NOT_FOUND: "The sentiment model is not available. Please try again later.",
  TOKENIZER_FAILURE: "The sentiment model is not available. Please try again later.",
  MODEL_LOAD_FAILURE: "The sentiment model is not available. Please try again later.",
  INVALID_REQUEST: "That text could not be analyzed. Check it and try again.",
  EMPTY_TEXT: "Enter some text before running the analysis.",
  TEXT_TOO_LONG: `Text is too long. Keep it under ${MAX_TEXT_LENGTH.toLocaleString()} characters.`,
  INFERENCE_FAILURE: "The model could not analyze this text. Please try again.",
  INTERNAL_ERROR: "Something went wrong on the server. Please try again.",
  BACKEND_UNAVAILABLE: "The analysis service is not available. Check that the backend is running.",
  NETWORK_FAILURE: "A network error occurred. Check your connection and try again.",
  TIMEOUT: "The request timed out. Please try again in a moment.",
  HTTP_ERROR: "The server returned an error. Please try again.",
  INVALID_RESPONSE: "The server returned an unexpected response. Please try again.",
};

const TITLES: Record<ErrorCode, string> = {
  MODEL_NOT_FOUND: "Service unavailable",
  TOKENIZER_FAILURE: "Service unavailable",
  MODEL_LOAD_FAILURE: "Service unavailable",
  INVALID_REQUEST: "Check your text",
  EMPTY_TEXT: "Check your text",
  TEXT_TOO_LONG: "Check your text",
  INFERENCE_FAILURE: "Analysis failed",
  INTERNAL_ERROR: "Analysis failed",
  BACKEND_UNAVAILABLE: "Service unavailable",
  NETWORK_FAILURE: "Connection problem",
  TIMEOUT: "Request timed out",
  HTTP_ERROR: "Analysis failed",
  INVALID_RESPONSE: "Analysis failed",
};

const RETRYABLE: ReadonlySet<ErrorCode> = new Set([
  "MODEL_NOT_FOUND",
  "TOKENIZER_FAILURE",
  "MODEL_LOAD_FAILURE",
  "INFERENCE_FAILURE",
  "INTERNAL_ERROR",
  "BACKEND_UNAVAILABLE",
  "NETWORK_FAILURE",
  "TIMEOUT",
  "HTTP_ERROR",
  "INVALID_RESPONSE",
]);

export class AppError extends Error {
  readonly code: ErrorCode;
  readonly status?: number;
  readonly title: string;
  readonly retryable: boolean;

  constructor(code: ErrorCode, status?: number) {
    super(USER_MESSAGES[code]);
    this.name = "AppError";
    this.code = code;
    this.status = status;
    this.title = TITLES[code];
    this.retryable = RETRYABLE.has(code);
  }
}

/** Kept for call sites that distinguish client-side checks from transport errors. */
export class ValidationError extends AppError {
  constructor(code: "EMPTY_TEXT" | "TEXT_TOO_LONG" = "EMPTY_TEXT") {
    super(code);
    this.name = "ValidationError";
  }
}

export class ApiError extends AppError {
  constructor(code: ErrorCode, status?: number) {
    super(code, status);
    this.name = "ApiError";
  }
}

export function errorFromUnknown(): AppError {
  return new AppError("INTERNAL_ERROR");
}

export function errorFromInvalidResponse(): ApiError {
  return new ApiError("INVALID_RESPONSE");
}

export function errorFromFetchFailure(caught: unknown): ApiError {
  if (isTimeout(caught)) {
    return new ApiError("TIMEOUT");
  }

  if (typeof navigator !== "undefined" && navigator.onLine === false) {
    return new ApiError("NETWORK_FAILURE");
  }

  const message = caught instanceof Error ? caught.message : "";
  if (/failed to fetch|networkerror|load failed|econnrefused/i.test(message)) {
    return new ApiError("BACKEND_UNAVAILABLE");
  }

  return new ApiError("NETWORK_FAILURE");
}

export function errorFromHttp(status: number, body: unknown): ApiError {
  const parsed = readStructuredError(body);
  if (parsed && isErrorCode(parsed.code)) {
    return new ApiError(parsed.code, status);
  }

  if (status === 422) return new ApiError("INVALID_REQUEST", status);
  if (status === 503) return new ApiError("BACKEND_UNAVAILABLE", status);
  if (status >= 500) return new ApiError("INTERNAL_ERROR", status);
  return new ApiError("HTTP_ERROR", status);
}

function isErrorCode(code: string): code is ErrorCode {
  return Object.prototype.hasOwnProperty.call(USER_MESSAGES, code);
}

function isTimeout(caught: unknown): boolean {
  if (typeof DOMException !== "undefined" && caught instanceof DOMException) {
    return caught.name === "TimeoutError" || caught.name === "AbortError";
  }
  return caught instanceof Error && (caught.name === "TimeoutError" || caught.name === "AbortError");
}

function looksLikeStackTrace(message: string): boolean {
  return /traceback|file "|\.py", line|runtimeerror/i.test(message);
}

function readStructuredError(body: unknown): { code: string; message: string } | null {
  if (typeof body !== "object" || body === null || !("error" in body)) {
    return null;
  }

  const error = (body as { error: unknown }).error;
  if (typeof error !== "object" || error === null) {
    return null;
  }

  const { code, message } = error as { code?: unknown; message?: unknown };
  if (typeof code !== "string" || typeof message !== "string") {
    return null;
  }
  if (looksLikeStackTrace(message)) {
    return { code, message: "" };
  }
  return { code, message };
}
