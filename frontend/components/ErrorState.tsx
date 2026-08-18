interface ErrorStateProps {
  message: string;
  onRetry?: () => void;
}

/** Inline error panel for a failed analysis. */
export function ErrorState({ message, onRetry }: ErrorStateProps) {
  return (
    <section
      role="alert"
      className="rounded-xl border border-danger/40 bg-danger-wash p-5"
    >
      <div className="flex items-start gap-3">
        <svg
          aria-hidden
          viewBox="0 0 20 20"
          fill="none"
          className="mt-0.5 size-5 shrink-0 text-danger"
        >
          <circle cx="10" cy="10" r="8.25" stroke="currentColor" strokeWidth="1.5" />
          <path
            d="M10 6v4.5M10 13.6v.4"
            stroke="currentColor"
            strokeWidth="1.5"
            strokeLinecap="round"
          />
        </svg>

        <div className="min-w-0 flex-1">
          <h2 className="text-sm font-semibold text-ink">Analysis failed</h2>
          <p className="mt-1 text-sm text-ink-secondary">{message}</p>

          {onRetry ? (
            <button
              type="button"
              onClick={onRetry}
              className="mt-3 rounded-lg border border-line bg-surface px-3 py-1.5 text-sm font-medium text-ink transition-colors hover:bg-surface-sunken focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ink"
            >
              Try again
            </button>
          ) : null}
        </div>
      </div>
    </section>
  );
}
