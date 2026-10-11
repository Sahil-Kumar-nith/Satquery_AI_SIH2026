# PHASE 13 — Gap closures identified during review

Four gaps were flagged against the official PS description text and
fixed. All are additive — no existing endpoint contracts were broken
(new fields only; extra keys pydantic just carries along).

## 1. NLP location/time extraction (`agent/nlp_extract.py`)
`classify_query()` in `agent/router.py` already classified *task type*
(VQA/grounding/change/etc.) but never surfaced *where* or *when* the
query was about. New `extract_location_time(query)`:
- Time: regex over explicit dates, years, month names, "between X and
  Y", and common relative phrases ("last year", "this monsoon").
- Location: a small India/ISRO-relevant gazetteer + a generic
  capitalized-phrase fallback (flagged `heuristic` vs `gazetteer` so
  the UI can show trust level).
- Wired into `POST /api/query` → new response fields
  `extracted_locations`, `extracted_time_expressions`,
  `nlp_extraction_note`, and a new execution-trace line.
- Rule-based on purpose (matches the router's own "do NOT over-engineer
  the agent" approach) — NOT a trained NER/temporal model. The map's
  drawn point/rectangle/polygon remains the authoritative location
  source for actually running a model.

## 2. Atmospheric + geometric correction, spectral-band checks (`preprocessing/correction.py`)
Wired into `POST /api/upload` → new `preprocessing_report` field on
every upload response:
- **Integrity**: constant/all-NaN raster detection beyond the existing
  "can rasterio open it" check.
- **Geometric**: verifies CRS + affine transform are present (i.e. the
  file is pre-georeferenced, standard for GeoTIFF products) and flags
  unusual pixel sizes. This is a *check*, not a re-registration — no
  ground-control-point correction or orthorectification is performed
  or fabricated.
- **Spectral**: reports band count/dtype and which band-dependent
  analyses (true color / NIR-style index) are structurally possible.
- **Atmospheric**: real Dark Object Subtraction (Chavez, 1988) — the
  simplest legitimate member of that technique family. Explicitly NOT
  a physically-modeled (6S/MODTRAN) or sensor-calibrated correction;
  skipped (and reported as such) for single-band/SAR uploads, since DOS
  is an optical-haze technique.

## 3 & 4. Visual evidence (confidence maps) + explicit source attribution (`evidence/confidence_map.py`, `api/query.py`)
Every task's `tool_output` now carries `confidence_map_png` (base64
PNG), `confidence_map_kind`, and `confidence_map_note`:
- `change_detection` / `change_vqa` / `optical_sar`: the confidence map
  IS the real per-pixel evidence array (diff z-score magnitude /
  optical-SAR agreement mask) that the scalar stats were computed from
  — not an approximation.
- `grounding`: a synthetic Gaussian heatmap centered on the returned
  box, scaled by the detector's match score — labeled explicitly as
  synthetic, not a learned saliency map.
- `vqa`: a flat color tint representing the answer-level confidence
  bucket, labeled explicitly as NOT a spatial map (VQA has no spatial
  output to visualize).

`QueryResponse` also gained a `sources` field: a structured list of
`{image_id, filename, modality, role}` for every image considered,
separate from the free-text `note` — a machine-readable answer to
"which uploaded file(s) is this evidence based on?"

## What this does NOT claim
- No trained NER/temporal-reasoning model (item 1).
- No physically-modeled atmospheric correction, no GCP-based geometric
  re-registration (item 2).
- No learned saliency/attention map for grounding or VQA (items 3/4) —
  only change/optical-SAR confidence maps are real per-pixel model
  evidence; the rest are clearly labeled visual aids.
