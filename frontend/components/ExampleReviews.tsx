"use client";

interface Example {
  label: string;
  text: string;
}

const POSITIVE_EXAMPLES: Example[] = [
  {
    label: "Loved the movie",
    text: "Absolutely loved this movie. The acting was incredible.",
  },
  {
    label: "Exceeded expectations",
    text: "The product quality exceeded my expectations.",
  },
  {
    label: "Fantastic experience",
    text: "Fantastic experience from beginning to end.",
  },
];

const NEGATIVE_EXAMPLES: Example[] = [
  {
    label: "Boring movie",
    text: "The movie was boring and far too long.",
  },
  {
    label: "Broke after two days",
    text: "The product broke after only two days.",
  },
  {
    label: "Disappointing",
    text: "Very disappointing experience.",
  },
];

const MIXED_EXAMPLES: Example[] = [
  {
    label: "Great acting, weak plot",
    text: "The cinematography and acting were excellent, but the plot was bland and the last hour dragged badly.",
  },
  {
    label: "Good product, poor support",
    text: "The headphones sound fantastic and feel well made, but customer support was slow and unhelpful when I needed a replacement cable.",
  },
  {
    label: "Tasty food, rude staff",
    text: "The meal itself was delicious and beautifully presented, yet the waiters were rude and we sat for forty minutes before anyone took our order.",
  },
  {
    label: "Funny but forgettable",
    text: "There were a few genuinely funny scenes, but overall the film felt messy, uneven, and not worth a second watch.",
  },
];

const LONG_EXAMPLES: Example[] = [
  {
    label: "Long positive review",
    text: `I went in with modest expectations and walked out completely won over. From the opening scene the film has a confidence that never lets up: the performances are layered, the dialogue feels lived-in rather than written, and the score swells at exactly the right moments without drowning the story.

What stayed with me most was the lead performance. Every glance and pause felt intentional, and the supporting cast matched that energy instead of competing with it. The pacing is tight, the ending is earned, and I left the theater wanting to talk about it for hours. This is easily one of the best films I have seen this year, and I would recommend it without hesitation.`,
  },
  {
    label: "Long negative review",
    text: `I wanted to like this product, but after two weeks of daily use I cannot recommend it to anyone. The build quality looks decent in photos, yet in person the plastic feels cheap, the buttons stick, and the battery already drains to empty in a few hours.

Worse than the hardware is the software. The companion app crashes on launch more often than it works, firmware updates failed twice, and support replied with a copy-paste script that did not address the actual defect. I requested a refund and was told to ship it back at my own expense. For the price, this is an embarrassing experience. Save your money.`,
  },
  {
    label: "Long mixed review",
    text: `This is a frustrating film because so much of it is genuinely good. The first act is gripping: sharp writing, a haunting score, and a central performance that could have carried a masterpiece. Several scenes are as moving as anything released this year, and the production design is stunning down to the smallest detail.

Then the second half unravels. Subplots pile up and go nowhere, the tone swings from intimate drama to noisy spectacle without warning, and a late twist undercuts the emotional work of the opening. I still admired large stretches of it, and I would not call it a failure, but I also cannot call it a success. It is a beautiful, ambitious movie that needed another draft and a braver edit.`,
  },
  {
    label: "Long sarcastic review",
    text: `Oh, this was an absolute triumph of wasted time. Nothing says premium quality like a device that overheats during setup and a user manual written as if punctuation were optional. The so-called smart features are a delight: they randomly disconnect, ignore every command, and then cheerfully announce that everything is working perfectly.

Customer service was equally impressive. After sitting on hold for fifty minutes I was told my issue was "user error," which I suppose is true if the error was buying the product in the first place. If you enjoy spending money to be insulted by inanimate objects, congratulations. You have found your masterpiece.`,
  },
];

const GROUPS: { heading: string; examples: Example[] }[] = [
  { heading: "Positive", examples: POSITIVE_EXAMPLES },
  { heading: "Negative", examples: NEGATIVE_EXAMPLES },
  { heading: "Mixed", examples: MIXED_EXAMPLES },
  { heading: "Longer reviews", examples: LONG_EXAMPLES },
];

interface ExampleReviewsProps {
  onSelect: (text: string) => void;
  disabled?: boolean;
}

export function ExampleReviews({ onSelect, disabled = false }: ExampleReviewsProps) {
  return (
    <section aria-labelledby="examples-heading" className="space-y-4">
      <h2
        id="examples-heading"
        className="text-xs font-semibold uppercase tracking-wider text-ink-muted"
      >
        Try an example
      </h2>

      {GROUPS.map(({ heading, examples }) => (
        <ExampleGroup
          key={heading}
          heading={heading}
          examples={examples}
          onSelect={onSelect}
          disabled={disabled}
        />
      ))}
    </section>
  );
}

function ExampleGroup({
  heading,
  examples,
  onSelect,
  disabled,
}: {
  heading: string;
  examples: Example[];
  onSelect: (text: string) => void;
  disabled: boolean;
}) {
  return (
    <div>
      <h3 className="text-xs font-medium text-ink-secondary">{heading}</h3>
      <ul className="mt-2 flex flex-wrap gap-2">
        {examples.map(({ label, text }) => (
          <li key={label}>
            <button
              type="button"
              onClick={() => onSelect(text)}
              disabled={disabled}
              title={text}
              className="max-w-full rounded-full border border-line bg-surface px-3.5 py-1.5 text-left text-sm text-ink-secondary transition-colors hover:border-ink-muted hover:text-ink focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ink disabled:cursor-not-allowed disabled:opacity-40"
            >
              {label}
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}
