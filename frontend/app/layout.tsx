import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Sentiment Classifier",
  description: "Sentiment predictions from a fine-tuned DistilBERT model",
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
