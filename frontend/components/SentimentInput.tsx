"use client";

import { COUNTER_WARNING_RATIO, MAX_TEXT_LENGTH } from "@/lib/constants";

interface SentimentInputProps {
  value: string;
  onChange: (value: string) => void;
  onAnalyze: () => void;
  onClear: () => void;
  isLoading: boolean;
}

export function SentimentInput({
  value,
  onChange,
  onAnalyze,
  onClear,
  isLoading,
}: SentimentInputProps) {
  const length = value.length;
  const isOverLimit = length > MAX_TEXT_LENGTH;
  const isNearLimit = length >= MAX_TEXT_LENGTH * COUNTER_WARNING_RATIO;
  const canAnalyze = value.trim().length > 0 && !isOverLimit && !isLoading;

  // Cmd/Ctrl+Enter submits, the convention for a textarea-driven form.
  const handleKeyDown = (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if ((event.metaKey || event.ctrlKey) && event.key === "Enter" && canAnalyze) {
      event.preventDefault();
      onAnalyze();
    }
  };

  return (
    <section className="rounded-xl border border-line bg-surface p-5 shadow-sm sm:p-6">
      <div className="flex items-baseline justify-between gap-4">
        <label htmlFor="sentiment-text" className="text-sm font-semibold text-ink">
          Your text
        </label>
        <span
          aria-live="polite"
          className={`text-xs tabular-nums ${
            isOverLimit
              ? "font-semibold text-danger"
              : isNearLimit
                ? "text-ink-secondary"
                : "text-ink-muted"
          }`}
        >
          {length.toLocaleString()} / {MAX_TEXT_LENGTH.toLocaleString()}
        </span>
      </div>

      <textarea
        id="sentiment-text"
        value={value}
        onChange={(event) => onChange(event.target.value)}
        onKeyDown={handleKeyDown}
        disabled={isLoading}
        rows={8}
        spellCheck
        aria-describedby="sentiment-text-hint"
        aria-invalid={isOverLimit}
        placeholder="Paste a review, comment, or any piece of writing to analyze…"
        className="mt-3 w-full resize-y rounded-lg border border-line bg-surface-sunken px-4 py-3 text-[15px] leading-relaxed text-ink placeholder:text-ink-muted focus:border-ink-muted focus:outline-none focus-visible:ring-2 focus-visible:ring-ink/15 disabled:opacity-60"
      />

      <div className="mt-4 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <p id="sentiment-text-hint" className="text-xs text-ink-muted">
          Press <kbd className="font-sans font-medium">⌘</kbd>/
          <kbd className="font-sans font-medium">Ctrl</kbd> +{" "}
          <kbd className="font-sans font-medium">Enter</kbd> to analyze
        </p>

        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={onClear}
            disabled={isLoading || length === 0}
            className="rounded-lg border border-line px-4 py-2 text-sm font-medium text-ink-secondary transition-colors hover:bg-surface-sunken focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ink disabled:cursor-not-allowed disabled:opacity-40"
          >
            Clear
          </button>

          <button
            type="button"
            onClick={onAnalyze}
            disabled={!canAnalyze}
            className="inline-flex items-center gap-2 rounded-lg bg-ink px-5 py-2 text-sm font-semibold text-page transition-opacity hover:opacity-90 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ink disabled:cursor-not-allowed disabled:opacity-40"
          >
            {isLoading ? (
              <>
                <span
                  aria-hidden
                  className="size-3.5 animate-spin rounded-full border-2 border-page/40 border-t-page"
                />
                Analyzing…
              </>
            ) : (
              "Analyze Sentiment"
            )}
          </button>
        </div>
      </div>
    </section>
  );
}
