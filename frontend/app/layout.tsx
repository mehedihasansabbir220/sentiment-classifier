import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "AI Sentiment Classifier",
  description: "Analyze text using a fine-tuned DistilBERT model",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
