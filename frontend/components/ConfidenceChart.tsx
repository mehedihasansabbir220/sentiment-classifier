"use client";

import { useEffect, useState } from "react";

import { formatPercent } from "@/lib/constants";
import type { Probabilities, Sentiment } from "@/lib/types";

interface ConfidenceChartProps {
  probabilities: Probabilities;
  /** The winning class, drawn at full weight; the other recedes. */
  sentiment: Sentiment;
}

const ROWS: { key: Sentiment; label: string; fill: string; dot: string }[] = [
  { key: "positive", label: "Positive", fill: "bg-positive", dot: "bg-positive" },
  { key: "negative", label: "Negative", fill: "bg-negative", dot: "bg-negative" },
];

/**
 * Horizontal probability bars, one per class, driven entirely by the values
 * the API returned.
 *
 * Categories are named on every row, so identity never rests on colour and no
 * legend box is needed. Bars are 10px, square at the baseline with a 4px
 * rounded data-end on a recessive track. Values are text, which doubles as the
 * table view. No charting library — the bars are two divs.
 */
export function ConfidenceChart({ probabilities, sentiment }: ConfidenceChartProps) {
  // Bars grow from zero on mount, so each new prediction animates in.
  const [grown, setGrown] = useState(false);
  useEffect(() => {
    const frame = requestAnimationFrame(() => setGrown(true));
    return () => cancelAnimationFrame(frame);
  }, []);

  return (
    <div>
      <h3 className="text-xs font-semibold uppercase tracking-wider text-ink-muted">
        Class probabilities
      </h3>

      <dl className="mt-4 space-y-4">
        {ROWS.map(({ key, label, fill, dot }, index) => {
          const value = probabilities[key] ?? 0;
          const isWinner = key === sentiment;
          // Keep a hairline of colour visible for near-zero probabilities.
          const width = value > 0 ? Math.max(value * 100, 1.5) : 0;

          return (
            <div
              key={key}
              className="group animate-result-in"
              style={{ animationDelay: `${120 + index * 70}ms` }}
            >
              <div className="flex items-baseline justify-between gap-3">
                <dt className="flex items-center gap-2 text-sm text-ink-secondary">
                  <span aria-hidden className={`size-2 shrink-0 rounded-full ${dot}`} />
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
                  className={`h-full rounded-r-[4px] transition-[width,opacity] duration-700 ease-out motion-reduce:transition-none ${fill} ${
                    isWinner ? "" : "opacity-45 group-hover:opacity-70"
                  }`}
                  style={{ width: `${grown ? width : 0}%` }}
                />
              </div>
            </div>
          );
        })}
      </dl>
    </div>
  );
}
