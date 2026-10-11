import type { Metadata } from "next";
import "./globals.css";
import { ThemeProvider } from "@/components/ThemeProvider";

// Using system fonts instead of next/font/google — avoids an external
// network dependency at build time (also one less thing to break on
// venue wifi during the demo).

export const metadata: Metadata = {
  title: "SatQuery AI",
  description: "Agentic vision-language assistant for remote-sensing imagery.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className="h-full antialiased" data-theme="mission">
      <body className="min-h-full flex flex-col font-sans">
        <ThemeProvider>{children}</ThemeProvider>
      </body>
    </html>
  );
}
