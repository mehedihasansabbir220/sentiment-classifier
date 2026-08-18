"use client";

import { useState } from "react";

import { ErrorState } from "@/components/ErrorState";
import { ExampleReviews } from "@/components/ExampleReviews";
import { LoadingState } from "@/components/LoadingState";
import { SentimentInput } from "@/components/SentimentInput";
import { SentimentResult } from "@/components/SentimentResult";
import { ApiError, ValidationError, predictSentiment } from "@/lib/api";
import type { PredictionResponse } from "@/lib/types";

type Status = "idle" | "loading" | "success" | "error";

/**
 * Owns the input/loading/result/error state machine so page.tsx stays a
 * layout-only server component. The API base URL lives in lib/constants,
 * never in a component.
 */
export function SentimentAnalyzer() {
  const [text, setText] = useState("");
  const [status, setStatus] = useState<Status>("idle");
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [resultKey, setResultKey] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const isLoading = status === "loading";

  const runAnalysis = async (input: string) => {
    // 1. Validate happens inside predictSentiment before any request is sent.
    // 2. Show the loading state.
    setStatus("loading");
    setError(null);
    setResult(null);

    try {
      // 3-4. Call FastAPI and receive the prediction.
      setResult(await predictSentiment(input));
      setResultKey((previous) => previous + 1);
      setStatus("success"); // 5-7. Result, confidence and probabilities render.
    } catch (caught) {
      // 8. Handle API errors.
      setError(
        caught instanceof ValidationError || caught instanceof ApiError
          ? caught.message
          : "Something went wrong while analyzing this text. Please try again.",
      );
      setStatus("error");
    }
  };

  const handleClear = () => {
    setText("");
    setResult(null);
    setError(null);
    setStatus("idle");
  };

  return (
    <div className="space-y-6">
      <ExampleReviews onSelect={setText} disabled={isLoading} />

      <SentimentInput
        value={text}
        onChange={setText}
        onAnalyze={() => void runAnalysis(text)}
        onClear={handleClear}
        isLoading={isLoading}
      />

      {status === "loading" ? <LoadingState /> : null}

      {status === "error" && error ? (
        <ErrorState message={error} onRetry={() => void runAnalysis(text)} />
      ) : null}

      {status === "success" && result ? (
        <SentimentResult key={resultKey} result={result} />
      ) : null}

      {status === "idle" ? (
        <p className="rounded-xl border border-dashed border-line px-5 py-8 text-center text-sm text-ink-muted">
          Results will appear here once you analyze some text.
        </p>
      ) : null}
    </div>
  );
}
