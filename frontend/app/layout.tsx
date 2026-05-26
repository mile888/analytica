import type { Metadata, Viewport } from "next";
import "./globals.css";
import { AppShell } from "@/components/AppShell";
import { ThemeProvider } from "@/components/ThemeProvider";
import { ToastProvider } from "@/components/ToastProvider";
import { listInvestigations } from "@/lib/api";

export const metadata: Metadata = {
  title: "Analytica Decision Workspace",
  description: "AI-powered analytical investigation workspace — natural-language data analysis, charts, and reports"
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  maximumScale: 5,
};

export const dynamic = "force-dynamic";
export const revalidate = 0;
export const fetchCache = "force-no-store";

export default async function RootLayout({ children }: { children: React.ReactNode }) {
  const recentInvestigations = await listInvestigations().catch(() => []);

  return (
    <html lang="en" suppressHydrationWarning>
      <body className="min-h-screen bg-slate-50 font-sans text-slate-950 antialiased dark:bg-slate-950 dark:text-slate-50">
        <ThemeProvider>
          <ToastProvider>
            <AppShell recentInvestigations={recentInvestigations}>{children}</AppShell>
          </ToastProvider>
        </ThemeProvider>
      </body>
    </html>
  );
}
