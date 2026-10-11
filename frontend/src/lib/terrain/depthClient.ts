// src/lib/terrain/depthClient.ts
//
// Frontend client for the 3D Terrain / Hologram feature.
// Wraps POST /api/process-depth (Next.js proxy -> terrain-worker) and
// decodes the base64 float32 buffer into a typed array Terrain3D/TerrainMesh
// can consume directly as heights.
//
// IMPORTANT (read before presenting to judges):
// This calls a generic monocular depth model (Depth-Anything V2), NOT an
// ISRO-PS-mandated remote-sensing model. Without ground-control points the
// output is *relative* depth (unitless, per-image), not calibrated
// elevation. Treat this as an optional visualization aid layered on top of
// whichever image is already loaded — it does not feed into VQA / change /
// optical-SAR results and must not be described to judges as one of the
// mandatory specialist models.

export interface ProcessDepthResponse {
  mode: "absolute" | "relative";
  calibration_method: "gcp" | "srtm" | "uncalibrated";
  calibration: {
    scale: number;
    offset: number;
    r_squared: number | null;
    n_points: number;
  };
  width: number;
  height: number;
  inference_ms: number;
  depth_preview_png_b64: string;
  depth_raw_f32_b64: string;
  geo: { crs: string | null; bounds: number[] | null } | null;
  dsm_geotiff_b64?: string | null;
}

export interface DepthResult extends Omit<ProcessDepthResponse, "depth_raw_f32_b64"> {
  /** Decoded heightmap, row-major, length = width * height. Meters if mode
   * === "absolute" (GCPs supplied), otherwise normalized relative depth. */
  heightmap: Float32Array;
}

/** Marker type kept for API-shape compatibility with the upstream
 * DepthWizard component; SatQuery's integration always passes []. */
export interface DetectedObject {
  label: string;
  score: number;
  center_x: number;
  center_y: number;
  inference_ms: number;
}

export async function processDepthImage(file: File, gcps?: unknown[]): Promise<DepthResult> {
  const formData = new FormData();
  formData.append("file", file);
  if (gcps && gcps.length > 0) {
    formData.append("gcps", JSON.stringify(gcps));
  }

  const res = await fetch("/api/process-depth", {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.error ?? `Request failed with status ${res.status}`);
  }

  const data = (await res.json()) as ProcessDepthResponse;

  return {
    ...data,
    heightmap: decodeFloat32Base64(data.depth_raw_f32_b64),
  };
}

function decodeFloat32Base64(b64: string): Float32Array {
  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i);
  }
  return new Float32Array(bytes.buffer);
}
