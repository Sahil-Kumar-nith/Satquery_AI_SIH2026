/**
 * Thin client for the SatQuery FastAPI backend.
 * Add typed functions here as each backend endpoint lands —
 * do not scatter raw fetch() calls through components.
 */
const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export interface HealthResponse {
  status: string;
  service: string;
  timestamp: string;
}

export async function getHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE_URL}/api/health`, { cache: "no-store" });
  if (!res.ok) {
    throw new Error(`Health check failed: ${res.status}`);
  }
  return res.json();
}

// ---- Phase 3: upload -------------------------------------------------

export type Modality = "optical" | "sar";
export type ImageRole = "single" | "before" | "after";

export interface UploadResponse {
  image_id: string;
  filename: string;
  modality: Modality;
  role: ImageRole;
  format: string;
  driver: string;
  width: number;
  height: number;
  band_count: number;
  dtype: string;
  crs: string | null;
  bounds: [number, number, number, number] | null;
  is_geospatial: boolean;
  size_bytes: number;
  // ---- Phase 14 (UI): surfaces the Phase 13 backend addition ----------
  preprocessing_report: PreprocessingReport | null;
}

// ---- Phase 14 (UI): preprocessing diagnostics from upload.py / -------
// preprocessing/correction.py (Phase 13 backend, previously unused by
// the frontend). See that module's "HONESTY CONTRACT" docstring — this
// is real integrity/georeferencing/spectral/DOS-atmospheric diagnostics,
// not a re-registration or physically-modeled correction.

export interface GeometricCheck {
  is_georeferenced: boolean;
  crs: string | null;
  pixel_size_x: number | null;
  pixel_size_y: number | null;
  note: string;
}

export interface SpectralCheck {
  band_count: number;
  dtype: string;
  supports_true_color: boolean;
  supports_nir_index: boolean;
  note: string;
}

export interface AtmosphericCorrection {
  applied: boolean;
  method: string;
  per_band_dark_object: number[];
  note: string;
}

export interface PreprocessingReport {
  integrity_ok: boolean;
  integrity_note: string;
  geometric: GeometricCheck | null;
  spectral: SpectralCheck | null;
  atmospheric: AtmosphericCorrection | null;
}

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

export async function uploadImage(
  file: File,
  modality: Modality,
  role: ImageRole,
): Promise<UploadResponse> {
  const form = new FormData();
  form.append("file", file);
  form.append("modality", modality);
  form.append("role", role);

  const res = await fetch(`${API_BASE_URL}/api/upload`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new ApiError(res.status, body?.detail ?? `Upload failed: ${res.status}`);
  }
  return res.json();
}

// ---- Phase 10: raster preview (for the Satellite View panel) ----------

/** Renders whatever raster was uploaded into a browser-displayable PNG.
 * Not calibrated imagery — same percentile-stretch disclosure as the
 * backend's preprocessing/raster_to_rgb.py. */
export function previewUrl(imageId: string): string {
  return `${API_BASE_URL}/api/upload/${imageId}/preview`;
}

// ---- Phase 4: query router --------------------------------------------

export type TaskType =
  | "vqa"
  | "captioning"
  | "grounding"
  | "change_detection"
  | "change_vqa"
  | "optical_sar";

export interface InputCheck {
  valid: boolean;
  issues: string[];
}

// ---- Phase 5: VQA specialist model output -----------------------------

export type ConfidenceLevel = "high" | "medium" | "insufficient";

// ---- Phase 14 (UI): confidence-map fields from evidence/confidence_map.py
// (Phase 13 backend). Present on every tool_output's raw dict but were
// never declared/rendered on the frontend. `kind` distinguishes real
// per-pixel model evidence from a synthetic/labelled visual aid — never
// present a "synthetic" or "flat_tint" map as if it were learned
// saliency. See ConfidenceMapCard in SatelliteView.tsx.
export type ConfidenceMapKind =
  | "change_diff"
  | "optical_sar_agreement"
  | "grounding_synthetic"
  | "flat_tint"
  | string;

export interface ConfidenceMapFields {
  confidence_map_png?: string | null;
  confidence_map_kind?: ConfidenceMapKind | null;
  confidence_map_note?: string | null;
}

export function getConfidenceMap(
  output: QueryResponse["tool_output"],
): Required<ConfidenceMapFields> | null {
  if (!output || typeof output !== "object") return null;
  const o = output as ConfidenceMapFields;
  if (!o.confidence_map_png) return null;
  return {
    confidence_map_png: o.confidence_map_png,
    confidence_map_kind: o.confidence_map_kind ?? "unknown",
    confidence_map_note: o.confidence_map_note ?? "",
  };
}

export interface VQAOutput extends ConfidenceMapFields {
  answer: string;
  confidence: number;
  confidence_level: ConfidenceLevel;
  confidence_note: string;
  model_id: string;
  adapted: boolean;
  latency_ms: number;
}

// ---- Phase 15: multi-method verification (change / optical+SAR only) ---
// NOT a second trained model — see backend/evidence/verification.py's
// docstring. Two independently-computed deterministic heuristics that can
// actually disagree with each other, surfaced here rather than hidden.
export interface VerificationOutput {
  method_a: string;
  method_a_level: string;
  method_b: string;
  method_b_level: string;
  agree: boolean;
  note: string;
}

export interface QueryResponse {
  query: string;
  task: TaskType;
  task_label: string;
  ps_requirement: string;
  classification_reason: string;
  images_considered: string[];
  input_check: InputCheck;
  selected_tool: string;
  status: "routed" | "blocked" | "insufficient_evidence" | "conflicting_evidence";
  execution_trace: string[];
  tool_output:
    | VQAOutput
    | GroundingOutput
    | ChangeDetectionOutput
    | ChangeVQAOutput
    | OpticalSarOutput
    | Record<string, unknown>
    | null;
  note: string;
  // ---- Phase 14 (UI): surfaces the Phase 13 backend additions ---------
  extracted_locations: string[];
  extracted_time_expressions: string[];
  nlp_extraction_note: string;
  sources: SourceRef[];
}

// ---- Phase 14 (UI): structured evidence attribution (Phase 13 backend,
// api/query.py SourceRef) — "which uploaded file(s) is this evidence
// based on?", separate from the free-text `note`.
export interface SourceRef {
  image_id: string;
  filename: string;
  modality: string;
  role: string;
}

export type NlpLocationSource = "gazetteer" | "heuristic" | "gazetteer+heuristic" | "none" | "unknown";

/** The backend never puts location_source on the QueryResponse itself —
 * it's only spelled out inside the "NLP Extraction: locations=... (kind),"
 * execution-trace line (see agent/nlp_extract.py + api/query.py). Parsed
 * client-side rather than duplicated, same pattern as SatelliteView.tsx's
 * inferActiveModality(). Purely cosmetic (trust badge) — never affects
 * what was actually extracted. */
export function inferLocationSource(executionTrace: string[]): NlpLocationSource {
  const line = executionTrace.find((l) => l.startsWith("NLP Extraction:"));
  if (!line) return "unknown";
  const match = line.match(/\(([a-z+]+)\)/);
  const val = match?.[1];
  if (val === "gazetteer" || val === "heuristic" || val === "gazetteer+heuristic" || val === "none") {
    return val;
  }
  return "unknown";
}

export function isVQAOutput(output: QueryResponse["tool_output"]): output is VQAOutput {
  return !!output && typeof output === "object" && "answer" in output && "confidence" in output;
}

// ---- Phase 6: grounding specialist model output ------------------------

export interface BoundingBox {
  x_min: number;
  y_min: number;
  x_max: number;
  y_max: number;
}

export interface GroundingOutput extends ConfidenceMapFields {
  phrase: string;
  found: boolean;
  box: BoundingBox | null;
  score: number;
  confidence_level: ConfidenceLevel;
  confidence_note: string;
  image_width: number;
  image_height: number;
  model_id: string;
  adapted: boolean;
  latency_ms: number;
  // Phase 16: which uploaded image this box was computed on — needed by
  // ChatPanel to carry the right previousImageId into a follow-up query.
  image_id?: string | null;
}

export function isGroundingOutput(output: QueryResponse["tool_output"]): output is GroundingOutput {
  return !!output && typeof output === "object" && "phrase" in output && "found" in output;
}

// ---- Phase 16: follow-up chain hop 2 — area estimate --------------------

export interface AreaEstimateOutput {
  target_phrase: string;
  box: BoundingBox;
  area_km2: number | null;
  area_note: string;
  method: string;
}

export function isAreaEstimateOutput(output: QueryResponse["tool_output"]): output is AreaEstimateOutput {
  return !!output && typeof output === "object" && "area_note" in output && "target_phrase" in output;
}

// ---- Phase 7: change detection / change-VQA specialist output ---------

export interface ChangeMap {
  changed_pixel_pct: number;
  change_level: "high" | "medium" | "low";
  change_note: string;
  largest_change_box: BoundingBox | null;
  width: number;
  height: number;
  method: string;
}

export interface ChangeDetectionOutput extends ConfidenceMapFields {
  description: string;
  change_map: ChangeMap;
  verification?: VerificationOutput;
}

export function isChangeDetectionOutput(
  output: QueryResponse["tool_output"],
): output is ChangeDetectionOutput {
  return !!output && typeof output === "object" && "description" in output && "change_map" in output;
}

export interface ChangeVQAOutput extends ConfidenceMapFields {
  answer: string;
  change_map: ChangeMap;
  vqa_confidence: number;
  vqa_confidence_level: ConfidenceLevel;
  model_id: string;
  adapted: boolean;
  verification?: VerificationOutput;
}

export function isChangeVQAOutput(output: QueryResponse["tool_output"]): output is ChangeVQAOutput {
  return !!output && typeof output === "object" && "answer" in output && "change_map" in output;
}

// ---- Phase 8: optical + SAR fusion specialist output -------------------

export interface OpticalSarOutput extends ConfidenceMapFields {
  target: "water" | "built_up";
  answer: string;
  agreement_pct: number;
  optical_only_pct: number;
  sar_only_pct: number;
  confidence_level: "high" | "medium" | "low";
  width: number;
  height: number;
  method: string;
  verification?: VerificationOutput;
}

export function isOpticalSarOutput(output: QueryResponse["tool_output"]): output is OpticalSarOutput {
  return !!output && typeof output === "object" && "agreement_pct" in output && "target" in output;
}

// ---- Phase 15/16: follow-up context. Passing the PREVIOUS turn's query
// and free-text answer lets a short referential query ("Where exactly?")
// lean on it — see backend/agent/followup.py. Phase 16 adds previousBox/
// previousImageId so hops 2 ("How much area?") and 3 ("Was this visible
// in SAR?") have the previous grounding result to work from, not just
// text. Nothing is persisted server-side; the caller (ChatPanel) is the
// source of truth for "previous turn".
export interface FollowupContext {
  previousQuery?: string;
  previousAnswer?: string;
  previousBox?: BoundingBox | null;
  previousImageId?: string | null;
}

export async function runQuery(
  query: string,
  imageIds: string[],
  followup?: FollowupContext,
): Promise<QueryResponse> {
  const res = await fetch(`${API_BASE_URL}/api/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query,
      image_ids: imageIds,
      previous_query: followup?.previousQuery ?? null,
      previous_answer: followup?.previousAnswer ?? null,
      previous_box: followup?.previousBox ?? null,
      previous_image_id: followup?.previousImageId ?? null,
    }),
  });

  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new ApiError(res.status, body?.detail ?? `Query failed: ${res.status}`);
  }
  return res.json();
}

// ---- Phase 10 (P0 fix): client-side task pre-classification ------------
//
// KNOWN GAP fixed here (see PHASE9.md §6 / SatQuery_User_Manual.txt):
// the chat panel used to send EVERY uploaded image with every query,
// which meant backend validate_inputs() would reject a request for
// having too many/wrong-role images unless the user manually cleared
// unrelated upload slots first. This mirrors backend/agent/router.py's
// deterministic keyword rules (kept in sync by hand — see that file if
// you change the rules there) so the chat panel can pick only the
// images relevant to what the query actually needs, automatically.
//
// This is a UX convenience only. The backend remains the source of
// truth for task classification and input validation — if this guess
// is wrong, the backend's own validation still runs and reports the
// real reason in the execution trace.

const _OPTICAL_SAR_KEYWORDS = [
  "optical and sar", "sar and optical", "combine optical", "fuse",
  "fusion", "using both", "both sensors", "radar and optical",
  "optical, sar", "together to identify",
];
const _CHANGE_KEYWORDS = [
  "chang", "increase", "decrease", "before and after", "differ",
  "compare these", "compared to", "over time",
];
const _GROUNDING_KEYWORDS = [
  "where is", "highlight", "locate", "point to", "show me the",
  "draw a box", "mark the",
];
const _CAPTIONING_KEYWORDS = [
  "describe", "caption", "summariz", "what does this show",
  "give a description", "overview of this image",
];

export type ClientGuessedTask =
  | "optical_sar"
  | "change"
  | "grounding"
  | "captioning"
  | "vqa";

/** Best-effort mirror of backend classify_query() — see comment above. */
export function classifyQueryClient(query: string): ClientGuessedTask {
  const q = query.toLowerCase().trim();
  if (_OPTICAL_SAR_KEYWORDS.some((k) => q.includes(k))) return "optical_sar";
  if (_CHANGE_KEYWORDS.some((k) => q.includes(k))) return "change";
  if (_GROUNDING_KEYWORDS.some((k) => q.includes(k))) return "grounding";
  if (_CAPTIONING_KEYWORDS.some((k) => q.includes(k))) return "captioning";
  return "vqa";
}

/**
 * Given every image currently uploaded (across all four slots) and a
 * query, return only the image_ids relevant to the guessed task —
 * matching what agent/registry.py's allowed_roles actually accepts.
 *
 *   optical_sar / change  -> need a specific pair (optical+sar / before+after)
 *   vqa / grounding / captioning -> single-role images only (optical
 *     and/or sar — both are sent when available so the Evidence
 *     Sufficiency Engine, Phase 9, can pick between them)
 */
export function selectImagesForQuery(
  query: string,
  images: UploadResponse[],
): { task: ClientGuessedTask; imageIds: string[] } {
  const task = classifyQueryClient(query);
  const single = images.filter((m) => m.role === "single");
  const before = images.filter((m) => m.role === "before");
  const after = images.filter((m) => m.role === "after");

  if (task === "optical_sar") {
    const optical = single.filter((m) => m.modality === "optical");
    const sar = single.filter((m) => m.modality === "sar");
    return { task, imageIds: [...optical, ...sar].map((m) => m.image_id) };
  }
  if (task === "change") {
    return { task, imageIds: [...before, ...after].map((m) => m.image_id) };
  }
  // vqa / grounding / captioning — single-image tasks (1 or 2 images:
  // optical and/or SAR of the same scene, for Phase 9 cloud->SAR switching)
  return { task, imageIds: single.map((m) => m.image_id) };
}
