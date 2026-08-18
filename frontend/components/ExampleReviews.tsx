"use client";

const EXAMPLES = [
  {
    label: "Glowing review",
    text: "I really enjoyed this movie. The performances were wonderful and the ending was perfect.",
  },
  {
    label: "Harsh review",
    text: "This was a boring, terrible waste of time. Awful acting and a dreadful script.",
  },
  {
    label: "Mixed review",
    text: "The cinematography was beautiful, but the plot was bland and the pacing dragged badly.",
  },
  {
    label: "Product feedback",
    text: "Arrived broken and the refund process was useless. Deeply disappointing experience.",
  },
];

interface ExampleReviewsProps {
  onSelect: (text: string) => void;
  disabled?: boolean;
}

export function ExampleReviews({ onSelect, disabled = false }: ExampleReviewsProps) {
  return (
    <section aria-labelledby="examples-heading">
      <h2
        id="examples-heading"
        className="text-xs font-semibold uppercase tracking-wider text-ink-muted"
      >
        Try an example
      </h2>

      <ul className="mt-3 flex flex-wrap gap-2">
        {EXAMPLES.map(({ label, text }) => (
          <li key={label}>
            <button
              type="button"
              onClick={() => onSelect(text)}
              disabled={disabled}
              title={text}
              className="rounded-full border border-line bg-surface px-3.5 py-1.5 text-sm text-ink-secondary transition-colors hover:border-ink-muted hover:text-ink focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ink disabled:cursor-not-allowed disabled:opacity-40"
            >
              {label}
            </button>
          </li>
        ))}
      </ul>
    </section>
  );
}
