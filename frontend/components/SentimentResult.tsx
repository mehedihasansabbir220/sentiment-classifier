import { ConfidenceChart } from "@/components/ConfidenceChart";
import { formatPercent } from "@/lib/constants";
import type { PredictionResponse } from "@/lib/types";

interface SentimentResultProps {
  result: PredictionResponse;
}

/**
 * Prediction panel: two stat tiles (sentiment, confidence) over the per-class
 * probability bars. Every value comes from the API response — nothing here is
 * hardcoded or derived from the input text.
 */
export function SentimentResult({ result }: SentimentResultProps) {
  const { sentiment, confidence, probabilities, text } = result;
  const isPositive = sentiment === "positive";

  const accentText = isPositive ? "text-positive" : "text-negative";
  const accentDot = isPositive ? "bg-positive" : "bg-negative";
  const accentWash = isPositive ? "bg-positive/10" : "bg-negative/10";

  return (
    <section
      aria-live="polite"
      className="animate-result-in overflow-hidden rounded-xl border border-line bg-surface shadow-sm"
    >
      {/* Hairline of the winning class across the top. */}
      <div aria-hidden className={`h-1 w-full ${accentDot}`} />

      <div className="p-5 sm:p-6">
        <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 sm:gap-6">
          <div
            className={`animate-result-in rounded-lg px-4 py-3.5 ${accentWash}`}
            style={{ animationDelay: "40ms" }}
          >
            <h2 className="text-xs font-semibold uppercase tracking-wider text-ink-muted">
              Sentiment
            </h2>
            <p className="mt-1.5 flex items-center gap-2.5">
              <span aria-hidden className={`size-2.5 shrink-0 rounded-full ${accentDot}`} />
              <span className="text-xl font-semibold uppercase tracking-wide text-ink sm:text-2xl">
                {sentiment}
              </span>
            </p>
          </div>

          <div
            className="animate-result-in rounded-lg bg-surface-sunken px-4 py-3.5"
            style={{ animationDelay: "90ms" }}
          >
            <h2 className="text-xs font-semibold uppercase tracking-wider text-ink-muted">
              Confidence
            </h2>
            <p className="mt-1 text-3xl font-semibold leading-tight text-ink sm:text-4xl">
              {formatPercent(confidence)}
            </p>
          </div>
        </div>

        <hr className="my-6 border-line" />

        <ConfidenceChart probabilities={probabilities} sentiment={sentiment} />

        <details className="mt-6 border-t border-line pt-4">
          <summary className="cursor-pointer text-xs font-medium text-ink-muted transition-colors hover:text-ink-secondary">
            Analyzed text
          </summary>
          <p className="mt-3 whitespace-pre-wrap break-words text-sm leading-relaxed text-ink-secondary">
            {text}
          </p>
        </details>
      </div>
    </section>
  );
}
