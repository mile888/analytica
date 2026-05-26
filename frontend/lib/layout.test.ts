import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

describe("root app styling contract", () => {
  it("keeps global Tailwind CSS and AppShell wired into the root layout", () => {
    const layout = readFileSync("app/layout.tsx", "utf8");
    const globals = readFileSync("app/globals.css", "utf8");
    const tailwindConfig = readFileSync("tailwind.config.ts", "utf8");

    expect(layout).toContain('import "./globals.css"');
    expect(layout).toContain("<ThemeProvider>");
    expect(layout).toContain("<ToastProvider>");
    expect(layout).toContain("<AppShell");
    expect(layout).toContain("font-sans");

    expect(globals).toContain("@tailwind base;");
    expect(globals).toContain("@tailwind components;");
    expect(globals).toContain("@tailwind utilities;");
    expect(globals).toContain("a {");

    expect(tailwindConfig).toContain("./app/**/*");
    expect(tailwindConfig).toContain("./components/**/*");
    expect(tailwindConfig).toContain("./lib/**/*");
  });
});
