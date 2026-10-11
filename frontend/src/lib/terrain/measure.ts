// src/lib/measure.ts
//
// The math engine MeasurePanel.tsx and Terrain3D.tsx both import from —
// this file didn't exist in the Qwen handoff, which is why every
// measurement tool (probe/distance/profile/slope) would have thrown a
// module-not-found error on compile, or silently no-op'd if stubbed.
//
// Core idea: TerrainMesh.tsx displaces a PlaneGeometry's Y values for
// VISUAL exaggeration (heightScale), but when the user clicks a point,
// we want to report the REAL underlying value (meters if absolute,
// unitless relative depth otherwise) — not the exaggerated display
// height. So this file independently re-samples the source heightmap
// at the clicked (x, z) world position, ignoring Y entirely.

import type { DepthResult } from "./depthClient";

/** Must match the planeSize TerrainMesh actually renders at. Single
 * source of truth — TerrainMesh imports this instead of hardcoding 20. */
export const PLANE_SIZE = 20;

/** Must match TerrainMesh's exaggeration formula (planeSize * 0.3) —
 * only relevant if you ever need to invert a displayed Y, which
 * heightAt() deliberately doesn't do (see module comment above). */
export const HEIGHT_EXAGGERATION_BASE = 0.3;

export type Tool = "probe" | "distance" | "profile" | "slope";

export interface Pick {
  /** World-space X on the rendered plane */
  x: number;
  /** World-space Y — the exaggerated DISPLAY height, from the raycast hit. Not used for math, kept for marker placement. */
  y: number;
  /** World-space Z on the rendered plane */
  z: number;
  /** Real underlying height value at this point (meters if absolute, unitless if relative) */
  h: number;
}

export interface TerrainMath {
  heightmap: Float32Array;
  sourceWidth: number;
  sourceHeight: number;
  mode: DepthResult["mode"];
  geo: DepthResult["geo"];
  /** Real-world meters per source-image pixel, only defined when mode === "absolute" and geo.bounds exists. */
  metersPerPixel: number | null;
  planeSize: number;
}

/**
 * Builds the shared context every measurement function needs. Call this
 * once per DepthResult (e.g. via useMemo in Terrain3D) and pass the
 * result into every function below.
 */
export function buildTerrainMath(
  heightmap: Float32Array,
  sourceWidth: number,
  sourceHeight: number,
  mode: DepthResult["mode"],
  geo: DepthResult["geo"],
  planeSize: number = PLANE_SIZE
): TerrainMath {
  let metersPerPixel: number | null = null;

  if (mode === "absolute" && geo?.bounds && geo.bounds.length === 4) {
    const [left, bottom, right, top] = geo.bounds;
    const widthMeters = Math.abs(right - left);
    const heightMeters = Math.abs(top - bottom);
    // Average the two axes — real projections aren't perfectly square,
    // but this is a reasonable approximation for a demo distance tool,
    // not a substitute for a proper geodesic calculation.
    const mppX = widthMeters / sourceWidth;
    const mppY = heightMeters / sourceHeight;
    metersPerPixel = (mppX + mppY) / 2;
  }

  return { heightmap, sourceWidth, sourceHeight, mode, geo, metersPerPixel, planeSize };
}

/** Bilinear sample of the raw heightmap at normalized (u, v) in [0, 1]. */
function sampleRaw(tm: TerrainMath, u: number, v: number): number {
  const { heightmap, sourceWidth: srcW, sourceHeight: srcH } = tm;
  const x = Math.min(Math.max(u, 0), 1) * (srcW - 1);
  const y = Math.min(Math.max(v, 0), 1) * (srcH - 1);
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const x1 = Math.min(x0 + 1, srcW - 1);
  const y1 = Math.min(y0 + 1, srcH - 1);
  const fx = x - x0;
  const fy = y - y0;

  const h00 = heightmap[y0 * srcW + x0];
  const h10 = heightmap[y0 * srcW + x1];
  const h01 = heightmap[y1 * srcW + x0];
  const h11 = heightmap[y1 * srcW + x1];

  const top = h00 * (1 - fx) + h10 * fx;
  const bottom = h01 * (1 - fx) + h11 * fx;
  return top * (1 - fy) + bottom * fy;
}

/** Converts a world-space (x, z) on the rendered plane to normalized (u, v). */
function worldToUV(tm: TerrainMath, x: number, z: number): { u: number; v: number } {
  const half = tm.planeSize / 2;
  const u = (x + half) / tm.planeSize;
  const v = (z + half) / tm.planeSize;
  return { u, v };
}

/** Real (non-exaggerated) height at a world-space (x, z). */
export function heightAt(tm: TerrainMath, x: number, z: number): number {
  const { u, v } = worldToUV(tm, x, z);
  return sampleRaw(tm, u, v);
}

/** Human-readable height string, unit-aware. */
export function formatHeight(tm: TerrainMath, h: number): string {
  if (tm.mode === "absolute") {
    return `${h.toFixed(2)} m`;
  }
  return `${(h * 100).toFixed(1)}% (relative)`;
}

/**
 * Local slope angle in degrees at (x, z), via finite differences.
 * Only meaningful in real units when metersPerPixel is known — returns
 * null for relative (non-georeferenced) data, matching what
 * MeasurePanel already checks for (`tm.metersPerPixel`).
 */
export function slopeAt(tm: TerrainMath, x: number, z: number): number | null {
  if (tm.metersPerPixel === null) return null;

  const { u, v } = worldToUV(tm, x, z);
  const eps = 1 / Math.max(tm.sourceWidth, tm.sourceHeight); // ~1 source pixel in uv space

  const hL = sampleRaw(tm, u - eps, v);
  const hR = sampleRaw(tm, u + eps, v);
  const hD = sampleRaw(tm, u, v - eps);
  const hU = sampleRaw(tm, u, v + eps);

  const stepMeters = eps * tm.sourceWidth * tm.metersPerPixel; // real distance covered by `eps` in u
  const dhdx = (hR - hL) / (2 * stepMeters);
  const dhdz = (hU - hD) / (2 * stepMeters);

  const gradMag = Math.sqrt(dhdx * dhdx + dhdz * dhdz);
  return (Math.atan(gradMag) * 180) / Math.PI;
}

export interface DistanceResult {
  /** Real horizontal distance in meters, or null if not georeferenced. */
  horizontalM: number | null;
  /** Difference in real height between the two points. */
  deltaH: number;
}

export function distanceMetrics(tm: TerrainMath, p1: Pick, p2: Pick): DistanceResult {
  const uv1 = worldToUV(tm, p1.x, p1.z);
  const uv2 = worldToUV(tm, p2.x, p2.z);

  let horizontalM: number | null = null;
  if (tm.metersPerPixel !== null) {
    const dxPixels = (uv2.u - uv1.u) * tm.sourceWidth;
    const dzPixels = (uv2.v - uv1.v) * tm.sourceHeight;
    const dxM = dxPixels * tm.metersPerPixel;
    const dzM = dzPixels * tm.metersPerPixel;
    horizontalM = Math.sqrt(dxM * dxM + dzM * dzM);
  }

  return { horizontalM, deltaH: p2.h - p1.h };
}

/**
 * Samples raw height at `steps` evenly-spaced points along the line
 * between p1 and p2, for the elevation-profile chart.
 */
export function profileBetween(tm: TerrainMath, p1: Pick, p2: Pick, steps: number = 48): number[] {
  const out: number[] = [];
  for (let i = 0; i < steps; i++) {
    const t = i / (steps - 1);
    const x = p1.x + (p2.x - p1.x) * t;
    const z = p1.z + (p2.z - p1.z) * t;
    out.push(heightAt(tm, x, z));
  }
  return out;
}