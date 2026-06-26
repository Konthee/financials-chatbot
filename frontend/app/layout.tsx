import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Financial QA Chatbot",
  description: "Grounded financial question answering for the Siametrics take-home task",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
