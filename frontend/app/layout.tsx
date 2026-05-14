import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "Analytica Decision Workspace",
  description: "Thin Next.js shell for the Analytica product backend"
};

const navItems = [
  { href: "/investigations", label: "Investigations" },
  { href: "/data-sources", label: "Data Sources" },
  { href: "/published-reports", label: "Published Reports" }
];

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <div className="min-h-screen lg:grid lg:grid-cols-[260px_1fr]">
          <aside className="border-r border-slate-200 bg-white px-5 py-6">
            <Link href="/investigations" className="block">
              <div className="text-lg font-semibold text-ink">Analytica</div>
              <div className="mt-1 text-sm text-muted">Decision Workspace</div>
            </Link>
            <nav className="mt-8 space-y-1">
              {navItems.map((item) => (
                <Link
                  key={item.href}
                  href={item.href}
                  className="block rounded-md px-3 py-2 text-sm font-medium text-slate-700 hover:bg-slate-100"
                >
                  {item.label}
                </Link>
              ))}
            </nav>
            <div className="mt-8 rounded-md border border-slate-200 bg-slate-50 p-3 text-xs leading-5 text-slate-600">
              Thin frontend shell over the existing FastAPI backend. No auth, no
              websocket, no heavy client state.
            </div>
          </aside>
          <main className="min-w-0 px-6 py-6 lg:px-8">{children}</main>
        </div>
      </body>
    </html>
  );
}
