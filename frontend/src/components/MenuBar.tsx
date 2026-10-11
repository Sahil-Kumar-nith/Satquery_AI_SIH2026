"use client";

import { useEffect, useState } from "react";
import SystemStatus from "./SystemStatus";
import ThemeSwitcher from "./ThemeSwitcher";

export type TabId = "dashboard" | "map" | "registry" | "benchmarks" | "report";

export const TABS: { id: TabId; label: string }[] = [
  { id: "dashboard", label: "Dashboard" },
  { id: "map", label: "Map" },
  { id: "registry", label: "Model Registry" },
  { id: "benchmarks", label: "Benchmarks" },
  { id: "report", label: "Report" },
];

interface MenuBarProps {
  activeTab: TabId;
  onTabChange: (tab: TabId) => void;
  readiness: { label: string; ready: boolean }[];
  onOpenTerrain?: () => void;
  terrainAvailable?: boolean;
}

export default function MenuBar({
  activeTab,
  onTabChange,
  readiness,
  onOpenTerrain,
  terrainAvailable = false,
}: MenuBarProps) {
  const [mobileNavOpen, setMobileNavOpen] = useState(false);

  return (
    <header className="border-b border-[var(--border)] bg-[var(--bg-panel)]">
      <div className="flex items-center gap-4 px-4 py-2.5 sm:px-6">
        {/* Brand */}
        <div className="flex items-baseline gap-2.5">
          <span className="text-lg leading-none">🛰</span>
          <h1 className="text-base font-semibold tracking-tight text-[var(--text)]">
            SATQUERY <span className="text-[var(--accent)]">AI</span>
          </h1>
          <span className="hidden text-xs text-[var(--text-faint)] lg:inline">
            Multimodal Remote Sensing Intelligence
          </span>
        </div>

        {/* Nav tabs — desktop */}
        <nav className="ml-2 hidden items-center gap-1 md:flex">
          {TABS.map((tab) => (
            <button
              key={tab.id}
              onClick={() => onTabChange(tab.id)}
              className={`rounded-md px-3 py-1.5 text-sm font-medium transition-colors ${
                activeTab === tab.id
                  ? "bg-[var(--accent-bg)] text-[var(--accent-text)]"
                  : "text-[var(--text-dim)] hover:bg-[var(--bg-elevated)] hover:text-[var(--text)]"
              }`}
            >
              {tab.label}
            </button>
          ))}
        </nav>

        <div className="flex-1" />

        {/* Readiness badges */}
        <div className="hidden items-center gap-3 text-[11px] font-medium text-[var(--text-faint)] xl:flex">
          {readiness.map((r) => (
            <span key={r.label} className="flex items-center gap-1">
              <span
                className={`h-1.5 w-1.5 rounded-full ${r.ready ? "bg-emerald-400" : "bg-[var(--border)]"}`}
              />
              {r.label}
            </span>
          ))}
        </div>

        {/* 3D Terrain / Hologram — optional visualization aid, see TerrainModal
            disclosure. Disabled until at least one image is uploaded. */}
        <button
          onClick={onOpenTerrain}
          disabled={!terrainAvailable}
          title={
            terrainAvailable
              ? "Generate a 3D terrain / hologram view of the active image"
              : "Upload an image first"
          }
          className={`hidden items-center gap-1.5 rounded-md border px-2.5 py-1.5 text-xs font-medium transition-colors sm:flex ${
            terrainAvailable
              ? "border-[var(--accent-strong)] text-[var(--accent-text)] hover:bg-[var(--accent-bg)]"
              : "cursor-not-allowed border-[var(--border)] text-[var(--text-faint)]"
          }`}
        >
          🧊 3D Terrain
        </button>

        <MissionClock />
        <ThemeSwitcher />
        <SystemStatus />

        {/* Mobile menu toggle */}
        <button
          onClick={() => setMobileNavOpen((o) => !o)}
          className="rounded-md border border-[var(--border)] px-2.5 py-1.5 text-xs text-[var(--text-dim)] md:hidden"
          aria-label="Toggle navigation menu"
        >
          ☰ Menu
        </button>
      </div>

      {/* Nav tabs — mobile */}
      {mobileNavOpen && (
        <nav className="flex flex-col gap-1 border-t border-[var(--border)] px-4 py-2 md:hidden">
          {TABS.map((tab) => (
            <button
              key={tab.id}
              onClick={() => {
                onTabChange(tab.id);
                setMobileNavOpen(false);
              }}
              className={`rounded-md px-3 py-2 text-left text-sm font-medium ${
                activeTab === tab.id
                  ? "bg-[var(--accent-bg)] text-[var(--accent-text)]"
                  : "text-[var(--text-dim)]"
              }`}
            >
              {tab.label}
            </button>
          ))}
          <button
            onClick={() => {
              onOpenTerrain?.();
              setMobileNavOpen(false);
            }}
            disabled={!terrainAvailable}
            className={`rounded-md px-3 py-2 text-left text-sm font-medium ${
              terrainAvailable ? "text-[var(--text-dim)]" : "cursor-not-allowed text-[var(--text-faint)]"
            }`}
          >
            🧊 3D Terrain
          </button>
        </nav>
      )}
    </header>
  );
}

/** Small "mission time" readout — purely cosmetic chrome to match the
 * ISRO/NASA mission-console reference look; not tied to any backend data. */
function MissionClock() {
  const [now, setNow] = useState<Date | null>(null);

  useEffect(() => {
    setNow(new Date());
    const id = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(id);
  }, []);

  if (!now) return null;

  const time = now.toLocaleTimeString("en-IN", { hour12: false });
  const date = now.toLocaleDateString("en-IN", { day: "2-digit", month: "short", year: "numeric" });

  return (
    <div className="hidden flex-col items-end leading-tight sm:flex">
      <span className="font-mono text-xs text-[var(--text)]">{time}</span>
      <span className="text-[10px] text-[var(--text-faint)]">{date}</span>
    </div>
  );
}
