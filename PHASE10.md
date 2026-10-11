# SatQuery AI — Phase 10 Report

**Scope:** Fix the two drawbacks flagged in `current_drawbacks.txt` / `PHASE9.md §6` (frontend
image-routing bug, no map/spatial evidence) and apply the Mission Control UI direction from
`uisuggestions.docx`.
**Status:** ✅ Both P0 items fixed. UI direction applied for the P0 subset (dashboard, viewer,
before/after slider, evidence panel, agent trace, layer control). P1/P2 items from that doc
(model registry page, benchmark page, report generation, follow-up chaining) are **not** built —
see §5.

---

## 1. What was actually broken (per your own project's documentation)

1. **`ChatPanel` sent every uploaded image with every query**, regardless of which upload slot
   it came from. `PHASE9.md §6` names this explicitly: *"the chat panel currently sends every
   image uploaded to the page with every query, not just the relevant ones."* Backend
   `validate_inputs()` (unchanged, still the source of truth) would then reject the request —
   e.g. asking a plain VQA question while a Before/After pair was also sitting in those slots
   produced `role 'before' not valid ... expected one of ['single']`, exactly as documented in
   `SatQuery_User_Manual.txt §7`. The user manual's workaround was "clear Before/After first."
2. **No map/image visualization.** `page.tsx` rendered a literal placeholder string: *"Map view
   — wired up in Phase 9 (map visualization)."* Grounding boxes were returned as raw pixel
   coordinates with a comment "(map overlay pending Phase 10)"; change maps and optical/SAR
   fusion had no visual output at all.

Both are fixed below. Nothing about the backend's actual model behavior, confidence numbers, or
honesty disclosures was changed — this phase is UI + one small backend addition (a preview
endpoint), not a re-architecture.

---

## 2. Fix 1 — task-aware image selection (P0)

### `backend/api/upload.py` — no change to routing logic itself
Not touched for this fix (the bug was purely client-side — the backend's `validate_inputs()`
was always correct, it just kept getting handed the wrong images).

### `frontend/src/lib/api.ts` — new `classifyQueryClient()` / `selectImagesForQuery()`
A deliberately simple, hand-mirrored copy of `backend/agent/router.py`'s keyword rules. Given
the query text and every currently-uploaded image (across all four slots), it returns only the
image_ids that slot actually applies to:

| Guessed task | Images sent |
|---|---|
| `optical_sar` (fusion keywords) | single-role Optical + single-role SAR |
| `change` (change/temporal keywords) | Before + After |
| `vqa` / `grounding` / `captioning` | single-role Optical and/or SAR (both, when both exist — so Phase 9's Evidence Sufficiency Engine can still pick between them) |

This is **client-side UX only** — a convenience so the demo doesn't require manually clearing
slots between questions. The backend remains the actual authority: its own `classify_query()` +
`validate_inputs()` still run on every request and are what's evaluated/trusted. If the client
guess and the backend's guess ever disagree, the backend's execution trace is what the user
sees and is correct.

### `frontend/src/components/ChatPanel.tsx`
- Now takes the full `uploaded` slot map (not a flat `imageIds` list) and calls
  `selectImagesForQuery()` per-query.
- Shows a one-line "Will use: <task> · N image(s)" hint above the input so the selection is
  visible before submitting, not a silent guess.
- If the guessed task has zero matching images, blocks the request client-side with a plain
  message instead of sending a request guaranteed to fail validation.

**Verified:** reproduced the exact bug against the live backend (`change_vqa` task given a
single `role=single` image → `status: "blocked"`, the same error text from the user manual),
confirming `selectImagesForQuery` would never construct that request in the first place because
it only pulls from Before/After slots for change queries.

---

## 3. Fix 2 — Satellite View panel (P0, replaces the placeholder)

### `backend/api/upload.py` — new `GET /api/upload/{image_id}/preview`
Renders whatever was uploaded (GeoTIFF, 1-band SAR, multi-band optical, PNG/JPEG) to a
browser-displayable PNG using the **same** `preprocessing/raster_to_rgb.py` percentile-stretch
pipeline the specialist models already use — so what's shown in the UI is honestly the same
pixels the models reasoned over, not a separately-touched-up "demo" image. Capped at 1600px on
the long edge (display only; model inference still runs on the source file). In-process cache
by `image_id`. No new dependency — reuses Pillow/Rasterio already in `requirements.txt`.

Smoke-tested directly against the running backend: uploaded a synthetic 128×128 GeoTIFF,
fetched `/api/upload/{id}/preview`, got back a valid 128×128 PNG.

### `frontend/src/components/SatelliteView.tsx` — new, replaces the placeholder `<section>`
Renders differently depending on the most recent query result's task:

- **Grounding** — the source image with the returned box drawn as an SVG-style overlay
  (`x_min/y_min/x_max/y_max` scaled from `image_width`/`image_height` to the rendered size).
  This is the literal "map overlay pending Phase 10" from `ChatPanel.tsx`'s old code — done.
- **Change detection / Change-VQA** — a drag (or click-and-drag) Before/After compare slider
  (`BeforeAfterSlider.tsx`, pure CSS `clip-path`, no new dependency) with the `largest_change_box`
  drawn on top so it's visible regardless of slider position, plus a changed-% bar colored by
  `change_level`.
- **Optical + SAR fusion** — both source images side by side, plus an agreement bar showing
  `agreement_pct` / `optical_only_pct` / `sar_only_pct` from the actual response.
- **Everything else / no result yet** — a thumbnail grid of whatever's currently uploaded, with
  the image the evidence engine actually used (inferred from the trace text containing "SAR
  activation" — see the code comment for why this is a heuristic, not a new backend field)
  highlighted.
- **Layer toggles** (Optical / SAR / Change / Grounding) in the panel header — matches
  `uisuggestions.docx`'s "Analysis Layers" request, scoped to what's actually renderable today
  (toggles visibility of the corresponding overlay/thumbnail; it does not yet control
  independent GIS layers on a real map, since there is no map/GIS library in this stack — see §5).

### `frontend/src/app/page.tsx`
- Wires `SatelliteView` in place of the placeholder, fed by the same `latestResult` state
  `ChatPanel` now reports upward via `onResult`.
- Mission Control header: `OPTICAL / SAR / GEO / AI` readiness dots (derived from what's
  actually uploaded — `GEO` lights up only if an uploaded file's `is_geospatial` flag is true,
  not just because *something* was uploaded).

---

## 4. Visual language change (uisuggestions.docx)

Per your own note in that file — *"Don't make it neon cyberpunk... ISRO/remote-sensing product
feel > gaming UI"* — the accent color moved from emerald-only to a restrained
navy/slate + cyan palette (cyan for primary actions/active states, emerald kept for
success/"ready" states, amber for warnings/medium-confidence, red for errors/blocked), applied
consistently across the header, chat, and Satellite View. Result cards were restructured
evidence-first (answer → confidence → spatial evidence → trace → caveats) instead of a flat
chat transcript, and the execution trace is now numbered (`01 02 03…`) rather than a plain
scrolling log, matching the "AGENT EXECUTION" mock in that file.

Result renderers were also added for `change_detection`, `change_vqa`, and `optical_sar` —
previously only `vqa` and `grounding` had dedicated cards in `ChatPanel.tsx`; the other three
task types fell through to the raw trace only.

---

## 5. What this phase deliberately does NOT do

Consistent with the master brief's 4-day-constraint discipline (don't build P1/P2 during a
crunch) and because none of these were named as drawbacks:

- **No real GIS/map library** (Leaflet/OpenLayers) — the Satellite View is an image/overlay
  viewer, not a georeferenced map with pan/zoom/CRS projection. Adding one is a reasonable next
  step once there's a reason to show multiple geographic scenes on one canvas; it isn't needed
  for the current single-scene-at-a-time demo flow and would cost real time for no PS-requirement
  benefit right now.
- **Model Registry page, Benchmark/eval page, Report generation, "Ask follow-up" chaining** —
  all explicitly P1/P2 in `uisuggestions.docx`'s own breakdown. Not built here; flagged, not
  forgotten.
- **The "which image did the evidence engine actually use" signal is still inferred from trace
  text on the frontend**, not returned as a structured field. If you want this load-bearing for
  anything beyond a UI highlight, add a `selected_modality` field to the Phase 9
  `SufficiencyDecision`/`VQAOutput` response instead of parsing trace strings — flagging this as
  the one heuristic in this phase that's UI-only and not something to demo as if it were
  structured output.
- **Remote-sensing adaptation status is unchanged** — still the #1 critical item from
  `current_drawbacks.txt`, and out of scope for a frontend/UI phase. Don't let a nicer UI distract
  from that being the next real technical task.

---

## 6. Verified working (this session)

- `npx next build` — compiles clean, zero type errors.
- `npx eslint src` — zero warnings/errors.
- Backend started live; `/api/upload` + new `/api/upload/{id}/preview` exercised end-to-end
  against a synthetic GeoTIFF — returns a valid PNG.
- Reproduced the exact `role 'before' not valid` bug directly against `/api/query` to confirm
  it's real, then confirmed `selectImagesForQuery()`'s task→slot mapping would never construct
  that request.

Not run in this session: the ML model paths (`models/vqa.py`, `grounding.py`, `change.py`,
`optical_sar.py`) — those require `torch`/`transformers`, which weren't installed for this
UI-focused smoke test. They are unchanged from Phase 5–9 and weren't touched by this phase.
