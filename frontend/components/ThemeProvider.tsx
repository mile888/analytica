"use client";

import { createContext, useContext, useEffect, useMemo, useState } from "react";

type ThemePreference = "system" | "light" | "dark";

const ThemeContext = createContext<{
  preference: ThemePreference;
  setPreference: (value: ThemePreference) => void;
}>({
  preference: "system",
  setPreference: () => {}
});

function applyTheme(preference: ThemePreference) {
  const systemDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  const resolved = preference === "system" ? (systemDark ? "dark" : "light") : preference;
  document.documentElement.classList.toggle("dark", resolved === "dark");
  document.documentElement.style.colorScheme = resolved;
}

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const [preference, setPreferenceState] = useState<ThemePreference>("system");

  useEffect(() => {
    const stored = window.localStorage.getItem("analytica-theme") as ThemePreference | null;
    const next = stored === "light" || stored === "dark" || stored === "system" ? stored : "system";
    setPreferenceState(next);
    applyTheme(next);

    const media = window.matchMedia("(prefers-color-scheme: dark)");
    const onChange = () => {
      if ((window.localStorage.getItem("analytica-theme") || "system") === "system") {
        applyTheme("system");
      }
    };
    media.addEventListener("change", onChange);
    return () => media.removeEventListener("change", onChange);
  }, []);

  const value = useMemo(
    () => ({
      preference,
      setPreference: (next: ThemePreference) => {
        window.localStorage.setItem("analytica-theme", next);
        setPreferenceState(next);
        applyTheme(next);
      }
    }),
    [preference]
  );

  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>;
}

export function useTheme() {
  return useContext(ThemeContext);
}
