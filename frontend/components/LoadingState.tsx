/** Skeleton shown while an analysis is in flight. Mirrors the result layout. */
export function LoadingState() {
  return (
    <section
      aria-busy="true"
      aria-live="polite"
      className="rounded-xl border border-line bg-surface p-6 shadow-sm"
    >
      <span className="sr-only">Analyzing sentiment…</span>

      <div className="flex items-center gap-3">
        <span
          aria-hidden
          className="size-4 animate-spin rounded-full border-2 border-line border-t-ink-secondary"
        />
        <p className="text-sm text-ink-secondary">Analyzing sentiment…</p>
      </div>

      <div aria-hidden className="mt-6 animate-pulse space-y-6">
        <div className="space-y-3">
          <div className="h-3 w-28 rounded bg-track" />
          <div className="h-12 w-40 rounded bg-track" />
        </div>
        <div className="space-y-4">
          <div className="h-3 w-32 rounded bg-track" />
          <div className="space-y-2">
            <div className="h-3 w-full rounded bg-track" />
            <div className="h-2.5 w-full rounded bg-track" />
          </div>
          <div className="space-y-2">
            <div className="h-3 w-full rounded bg-track" />
            <div className="h-2.5 w-full rounded bg-track" />
          </div>
        </div>
      </div>
    </section>
  );
}
