import { formatPercent } from "@/lib/constants";
import type { Probabilities, Sentiment } from "@/lib/types";

interface ConfidenceChartProps {
  probabilities: Probabilities;
  /** The winning class, drawn at full weight; the other recedes. */
  sentiment: Sentiment;
}

const ROWS: { key: Sentiment; label: string; bar: string; dot: string }[] = [
  { key: "positive", label: "Positive", bar: "bg-positive", dot: "bg-positive" },
  { key: "negative", label: "Negative", bar: "bg-negative", dot: "bg-negative" },
];

/**
 * Horizontal probability bars, one per class.
 *
 * Categories are named on every row, so identity never rests on colour alone
 * and no legend box is needed. Bars are 10px, square at the baseline with a
 * 4px rounded data-end, on a recessive track. Values are text, which doubles
 * as the table view.
 */
export function ConfidenceChart({ probabilities, sentiment }: ConfidenceChartProps) {
  return (
    <div>
      <h3 className="text-xs font-semibold uppercase tracking-wider text-ink-muted">
        Class probabilities
      </h3>

      <dl className="mt-4 space-y-4">
        {ROWS.map(({ key, label, bar, dot }) => {
          const value = probabilities[key] ?? 0;
          const isWinner = key === sentiment;

          return (
            <div key={key} className="group">
              <div className="flex items-baseline justify-between gap-3">
                <dt className="flex items-center gap-2 text-sm text-ink-secondary">
                  <span aria-hidden className={`size-2 rounded-full ${dot}`} />
                  <span className={isWinner ? "font-semibold text-ink" : undefined}>
                    {label}
                  </span>
                </dt>
                <dd
                  className={`text-sm tabular-nums ${
                    isWinner ? "font-semibold text-ink" : "text-ink-secondary"
                  }`}
                >
                  {formatPercent(value)}
                </dd>
              </div>

              <div
                className="mt-2 h-2.5 w-full rounded-r-[4px] bg-track"
                role="img"
                aria-label={`${label}: ${formatPercent(value)}`}
                title={`${label} — ${formatPercent(value)}`}
              >
                <div
                  className={`h-full rounded-r-[4px] transition-[width] duration-500 ease-out ${bar} ${
                    isWinner ? "" : "opacity-45 group-hover:opacity-70"
                  }`}
                  style={{ width: `${Math.max(value * 100, value > 0 ? 1.5 : 0)}%` }}
                />
              </div>
            </div>
          );
        })}
      </dl>
    </div>
  );
}
