# PHASE 14 — UI-only: surface what Phase 13's backend already computes

Scope discipline for this phase: **no backend changes, no new endpoints,
no new API contracts.** Phase 13 (`PHASE13.md`) added four fields to
existing responses — `confidence_map_png/kind/note` on every
`tool_output`, `preprocessing_report` on every upload, and
`extracted_locations` / `extracted_time_expressions` /
`nlp_extraction_note` / `sources` on every query response — but the
frontend's `lib/api.ts` types predated Phase 13 and never declared
them, so none of it reached the UI. Phase 14 is purely that: render
data the backend was already sending.

Comparison inputs for this phase, per request:
- `satquery-full` — diffed against phase13's frontend; the two are
  functionally identical (a `depthwizard`→`terrain` rename plus one
  extra Next.js API route). No unique UI ideas were pulled from it —
  documented here so it isn't re-diffed next time.
- `workspace` (standalone Vite prototype) — the `pulse-dot` keyframe
  animation was ported from its `index.css` for the
  insufficient/conflicting-evidence status badges. Its other components
  (CompareSlider, InfoTabs, ReportTab, MapTab) cover ground phase13's
  Next.js app already has (BeforeAfterSlider, ModelRegistryView /
  BenchmarksView / ReportView, GeoMap) at comparable or greater
  maturity, so nothing else was ported wholesale.

## 1. Confidence Map layer (`SatelliteView.tsx`)

Added a 5th "Confidence" toggle to the existing Analysis Layers row
(`optical / sar / change / grounding / confidence`) — this was a
labelled checkbox with nothing behind it in `uisuggestions.docx`'s
original mockup. Now backed by the real `confidence_map_png` on
whatever `tool_output` the current result carries.

New `ConfidenceMapCard` renders the PNG plus an explicit badge:
- **MODEL EVIDENCE** (green) — `change_diff` / `optical_sar_agreement`:
  the map IS the real per-pixel array the scalar stats were computed
  from.
- **SYNTHETIC AID** (amber) — `grounding_synthetic`: a Gaussian heatmap
  centered on the returned box, scaled by match score. Explicitly *not*
  a learned saliency map.
- **NOT SPATIAL** (neutral) — `flat_tint`: VQA has no spatial output;
  this is a flat color swatch for the answer-level confidence bucket
  only.

This mirrors evidence/confidence_map.py's own honesty contract
(PHASE13.md §3&4) — the UI never presents a synthetic or flat aid as if
it were learned model evidence.

## 2. Preprocessing diagnostics popover (`UploadSlot.tsx`)

Each upload slot's pill gained a small "ⓘ" button (visible once a file
is uploaded) that opens a popover showing `preprocessing_report`:
- Integrity (constant/NaN raster check)
- Georeferencing (CRS + pixel size present, or explicitly flagged
  absent — a *check*, never a fabricated transform)
- Spectral bands (count, dtype, true-color / NIR-index support)
- Atmospheric correction (Dark Object Subtraction applied or skipped,
  with the "not a physically-modeled correction" qualifier intact)

Purely a read-only display of what the backend already returned —
no new validation logic, no re-computation.

## 3. Query NLP extraction + source attribution (`ChatPanel.tsx`)

Two new rows on every result card, before the answer:

- **Location/time chips** (`📍 Assam`, `🗓 July 2026`, …) — plus a trust
  badge (`gazetteer match` / `heuristic guess` / …) parsed client-side
  from the `"NLP Extraction: ... (kind)"` execution-trace line, since
  `location_source` itself isn't a structured response field. This
  mirrors the existing `inferActiveModality()` pattern already used in
  `SatelliteView.tsx` (Phase 10) rather than inventing a new
  client-inference approach. Renders nothing if the query had no
  detectable location/time text — the map selection remains the
  authoritative scope source, exactly as `nlp_extract.py` documents.
- **Sources row** — the structured `{filename, modality, role}` list
  from the new `sources` field, distinct from the free-text `note`.

A small `ConfidenceMapHint` line ("🗺 Confidence map available — toggle
Confidence in Satellite View") was also added so the chat panel stays
text/evidence-summary only and doesn't duplicate the image render.

## 4. Small polish

- Ported the `pulse-dot` keyframe from the `workspace` prototype's
  `index.css` onto the `INSUFFICIENT EVIDENCE` / `CONFLICTING EVIDENCE`
  status badges only — a restrained "this needs attention" cue, not a
  decorative effect (see `uisuggestions.docx`: "don't make it neon
  cyberpunk").

## Incidental fix (not in original scope, but blocked `next build`)

`components/terrain/Terrain3D.tsx` passed a functional updater to a
`setPicks` prop typed as `(picks: Pick[]) => void` (doesn't accept
updater functions) — a pre-existing type error in the Phase 13
baseline that made `next build` fail entirely (confirmed by building
`satquery-phase13` unmodified: same two errors). Fixed by computing
`next` from the `picks` prop directly instead of a functional update.
Behavior-identical, typing-only change.

## Verified

- `npx tsc --noEmit` — clean except one pre-existing, out-of-scope
  `LayoutProps<"/">` error that is a bare-`tsc` artifact only (Next's
  own build generates that type; `next build` does not report it).
- `npx eslint` on all four touched files — clean (one pre-existing,
  untouched-by-this-phase warning in `ChatPanel.tsx`'s existing
  `pendingInsert` effect, confirmed present in the Phase 13 baseline
  too).
- `npx next build` — succeeds, 0 errors, matches Phase 13's route
  output (`/`, `/api/gibs-snapshot`, `/api/process-depth`).

## What this does NOT do

- No backend files were touched. Every field rendered here was already
  being computed and returned by Phase 13's backend.
- No new endpoints, no new response fields, no schema changes.
- Does not re-litigate or "improve" Phase 13's honesty qualifiers
  (DOS-not-6S, check-not-reregistration, synthetic-not-saliency) — it
  surfaces them verbatim.
- Does not port `workspace`'s CompareSlider/InfoTabs/ReportTab/MapTab —
  phase13's Next.js app already covers that ground (BeforeAfterSlider,
  ReportView, ModelRegistryView/BenchmarksView, GeoMap) at comparable
  maturity; re-implementing them would be scope creep for a "just UI"
  phase, not a genuine gap.
