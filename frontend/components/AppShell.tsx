"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Investigation } from "@/lib/api";
import { ThemeToggle } from "@/components/ThemeToggle";
import { formatDate } from "@/components/ui";
import { productInvestigations } from "@/lib/investigations";

const navItems = [
  { href: "/", label: "Dashboard" },
  { href: "/investigations", label: "Investigations" },
  { href: "/data-sources", label: "Datasets" },
  { href: "/reports", label: "Reports" }
];

export function AppShell({
  children,
  recentInvestigations
}: {
  children: React.ReactNode;
  recentInvestigations: Investigation[];
}) {
  const pathname = usePathname();
  const crumbs = buildBreadcrumbs(pathname);
  const visibleRecentInvestigations = productInvestigations(recentInvestigations).slice(0, 3);

  return (
    <div className="min-h-screen lg:grid lg:grid-cols-[280px_1fr]">
      <aside className="no-print border-r border-slate-200 bg-white px-5 py-6 dark:border-slate-800 dark:bg-slate-950">
        <Link href="/" className="block">
          <div className="text-lg font-semibold text-ink dark:text-slate-50">Analytica</div>
          <div className="mt-1 text-sm text-muted dark:text-slate-400">Analytical Investigation Agent</div>
        </Link>
        <nav className="mt-8 space-y-1">
          {navItems.map((item) => {
            const active = pathname === item.href || pathname.startsWith(`${item.href}/`);
            const isDashboard = item.href === "/";
            const isActive = isDashboard ? pathname === "/" : active;
            return (
              <Link
                key={item.href}
                href={item.href}
                className={`block rounded-md px-3 py-2 text-sm font-medium transition ${
                  isActive
                    ? "bg-slate-900 text-white dark:bg-slate-100 dark:text-slate-950"
                    : "text-slate-700 hover:bg-slate-100 dark:text-slate-300 dark:hover:bg-slate-900"
                }`}
              >
                {item.label}
              </Link>
            );
          })}
        </nav>

        <div className="mt-8">
          <div className="text-xs font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400">
            Recent investigations
          </div>
          <div className="mt-3 space-y-2">
            {visibleRecentInvestigations.length ? (
              visibleRecentInvestigations.map((item) => (
                <Link
                  key={item.investigation_id}
                  href={`/investigations/${item.investigation_id}`}
                  className="block rounded-md border border-slate-200 bg-slate-50/70 p-3 text-xs hover:bg-slate-100 dark:border-slate-800 dark:bg-slate-900/70 dark:hover:bg-slate-800"
                >
                  <div className="line-clamp-2 font-medium text-slate-800 dark:text-slate-100">{item.title}</div>
                  <div className="mt-1 text-slate-500 dark:text-slate-400">{formatDate(item.updated_at)}</div>
                </Link>
              ))
            ) : (
              <p className="rounded-md border border-dashed border-slate-200 p-3 text-xs text-slate-500 dark:border-slate-800 dark:text-slate-400">
                No recent investigations yet.
              </p>
            )}
          </div>
        </div>
      </aside>

      <div className="min-w-0">
        <header className="no-print sticky top-0 z-10 border-b border-slate-200 bg-slate-50/90 px-6 py-3 backdrop-blur dark:border-slate-800 dark:bg-slate-950/85 lg:px-8">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <nav className="text-sm text-slate-500 dark:text-slate-400">
              {crumbs.map((crumb, index) => (
                <span key={`${crumb.href}-${crumb.label}`}>
                  {index > 0 ? <span className="mx-2">/</span> : null}
                  {index === crumbs.length - 1 ? (
                    <span className="font-medium text-slate-900 dark:text-slate-100">{crumb.label}</span>
                  ) : (
                    <Link href={crumb.href} className="hover:text-slate-900 dark:hover:text-slate-100">
                      {crumb.label}
                    </Link>
                  )}
                </span>
              ))}
            </nav>
            <div className="hidden sm:block">
              <ThemeToggle />
            </div>
          </div>
        </header>
        <main className="min-w-0 px-4 py-6 sm:px-6 lg:px-8">{children}</main>
      </div>
    </div>
  );
}

function buildBreadcrumbs(pathname: string) {
  const parts = pathname.split("/").filter(Boolean);
  if (!parts.length) return [{ href: "/", label: "Dashboard" }];

  const labels: Record<string, string> = {
    "": "Dashboard",
    investigations: "Investigations",
    "data-sources": "Datasets",
    reports: "Reports"
  };

  return parts.map((part, index) => {
    const href = `/${parts.slice(0, index + 1).join("/")}`;
    const label = labels[part] || compactId(part);
    return { href, label };
  });
}

function compactId(value: string) {
  return value.length > 14 ? `${value.slice(0, 10)}...` : value;
}
