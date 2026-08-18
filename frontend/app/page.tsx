import { SentimentAnalyzer } from "@/components/SentimentAnalyzer";

export default function Home() {
  return (
    <main className="mx-auto min-h-dvh w-full max-w-3xl px-5 py-12 sm:px-8 sm:py-16">
      <header>
        <h1 className="text-3xl font-semibold tracking-tight text-ink sm:text-4xl">
          AI Sentiment Classifier
        </h1>
        <p className="mt-3 text-base text-ink-secondary">
          Analyze text using a fine-tuned DistilBERT model.
        </p>
      </header>

      <div className="mt-10">
        <SentimentAnalyzer />
      </div>

      <footer className="mt-16 border-t border-line pt-6 text-xs text-ink-muted">
        Fine-tuned DistilBERT · binary sentiment classification
      </footer>
    </main>
  );
}
