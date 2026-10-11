export type ThemeId = "mission" | "orbit" | "monsoon";

export interface ThemeMeta {
  id: ThemeId;
  name: string;
  tagline: string;
  swatch: [string, string, string]; // bg, panel, accent — for the picker preview dot
}

/** Three UI directions on the same layout/data — pick one now, swap later.
 * Not different features: same Mission Control shell, different visual
 * identity. See globals.css for the actual token values. */
export const THEMES: ThemeMeta[] = [
  {
    id: "mission",
    name: "Mission Control",
    tagline: "Navy + cyan dashboard",
    swatch: ["#020617", "#0f172a", "#22d3ee"],
  },
  {
    id: "orbit",
    name: "Orbit",
    tagline: "Deep-space tracking console",
    swatch: ["#000000", "#0a0a0a", "#f5a623"],
  },
  {
    id: "monsoon",
    name: "SAR Monsoon",
    tagline: "Radar-green, tuned to the flood demo",
    swatch: ["#030b08", "#071913", "#34d399"],
  },
];

export const DEFAULT_THEME: ThemeId = "mission";
export const THEME_STORAGE_KEY = "satquery.theme";
