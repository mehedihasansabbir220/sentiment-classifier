import { ConfidenceChart } from "@/components/ConfidenceChart";
import { formatPercent } from "@/lib/constants";
import type { PredictionResponse } from "@/lib/types";

interface SentimentResultProps {
  result: PredictionResponse;
}

/** Hero confidence figure plus the per-class breakdown. */
export function SentimentResult({ result }: SentimentResultProps) {
  const { sentiment, confidence, probabilities, text } = result;
  const isPositive = sentiment === "positive";

  return (
    <section
      aria-live="polite"
      className="rounded-xl border border-line bg-surface p-6 shadow-sm"
    >
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <h2 className="text-xs font-semibold uppercase tracking-wider text-ink-muted">
            Predicted sentiment
          </h2>

          <div className="mt-3 flex items-center gap-2.5">
            <span
              aria-hidden
              className={`size-3 rounded-full ${isPositive ? "bg-positive" : "bg-negative"}`}
            />
            <span className="text-2xl font-semibold capitalize text-ink">{sentiment}</span>
          </div>
        </div>

        <div className="text-right">
          <h2 className="text-xs font-semibold uppercase tracking-wider text-ink-muted">
            Confidence
          </h2>
          <p className="mt-2 text-5xl font-semibold leading-none text-ink">
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
    </section>
  );
}
