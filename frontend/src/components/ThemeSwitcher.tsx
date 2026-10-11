"use client";

import { useEffect, useRef, useState } from "react";
import { useTheme } from "./ThemeProvider";
import { THEMES } from "@/lib/theme";

export default function ThemeSwitcher() {
  const { theme, setTheme } = useTheme();
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const active = THEMES.find((t) => t.id === theme) ?? THEMES[0];

  useEffect(() => {
    function onClick(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    }
    document.addEventListener("mousedown", onClick);
    return () => document.removeEventListener("mousedown", onClick);
  }, []);

  return (
    <div ref={ref} className="relative">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        title="Switch console theme"
        className="flex items-center gap-2 rounded-md border border-[var(--border)] bg-[var(--bg-panel)] px-2.5 py-1.5 text-xs text-[var(--text-dim)] hover:border-[var(--accent)] hover:text-[var(--text)]"
      >
        <span
          className="h-3 w-3 rounded-full border border-[var(--border)]"
          style={{ background: active.swatch[2] }}
        />
        <span className="hidden sm:inline">{active.name}</span>
        <span className="text-[var(--text-faint)]">▾</span>
      </button>

      {open && (
        <div className="absolute right-0 top-[calc(100%+6px)] z-30 w-64 overflow-hidden rounded-lg border border-[var(--border)] bg-[var(--bg-panel)] shadow-xl">
          <div className="border-b border-[var(--border)] px-3 py-2 text-[10px] font-semibold tracking-wide text-[var(--text-faint)]">
            CONSOLE THEME — preview, wire in later
          </div>
          {THEMES.map((t) => (
            <button
              key={t.id}
              onClick={() => {
                setTheme(t.id);
                setOpen(false);
              }}
              className={`flex w-full items-center gap-3 px-3 py-2.5 text-left text-sm transition-colors hover:bg-[var(--bg-elevated)] ${
                t.id === theme ? "bg-[var(--bg-elevated)]" : ""
              }`}
            >
              <span className="flex overflow-hidden rounded border border-[var(--border)]">
                {t.swatch.map((c, i) => (
                  <span key={i} className="block h-5 w-4" style={{ background: c }} />
                ))}
              </span>
              <span className="flex-1">
                <span className="block text-[var(--text)]">{t.name}</span>
                <span className="block text-[11px] text-[var(--text-faint)]">{t.tagline}</span>
              </span>
              {t.id === theme && <span className="text-[var(--accent)]">✓</span>}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
