# SatQuery AI — Phase 12 Report

**Scope:** Close the three gaps PHASE11.md §5 named as open, plus the map gap named
in README.md §12 / master brief §4E:
1. SAR-before/SAR-after upload slots + real `insufficient_sar_activated` branch for
   change tasks.
2. CONFLICTING evidence state.
3. Real Leaflet/OpenLayers map with click-point / draw-rectangle / draw-polygon
   selection.
4. Attempt to actually run the BigEarthNet LoRA adaptation.

**Status:** 1–3 done and verified. 4 partially done — verified honestly, not faked
(see §4 below for exactly what that means).

---

## 1. SAR-before/SAR-after fallback for change tasks

**The gap:** PHASE11.md generalized the Evidence Sufficiency Engine's cloud check to
change tasks, but could only ever *detect* cloud obstruction on a Before/After pair
— there was nothing to fail over to, because the upload scheme had exactly one
Before slot and one After slot, both hard-coded optical.

**What changed:**
- `frontend/src/app/page.tsx` — two new upload slots, **SAR Before** / **SAR After**
  (`role: before/after`, `modality: sar`), alongside the existing four.
- `backend/agent/registry.py` — `change_detection`/`change_vqa` `max_images` raised
  2→4 so an optical pair AND a SAR pair can be supplied in the same request.
- `backend/evidence/sufficiency.py` — `decide_temporal_evidence()` now accepts
  optional `sar_before_path`/`sar_after_path` and returns a real
  `insufficient_sar_activated` outcome, mirroring `decide_evidence()`'s shape for
  the single-image case. **SAR activation requires BOTH SAR images** — a single SAR
  image cannot support a bi-temporal comparison, and supplying only one is treated
  the same as "no fallback available," with an explicit trace line saying so rather
  than silently ignoring the half-supplied pair.
- `backend/api/query.py` — `_temporal_paths_by_modality()` splits whatever
  Before/After images were supplied by modality; `_run_change_detection` /
  `_run_change_vqa` now route to whichever pair the engine actually selected, and
  the response `note` says explicitly when the SAR pair was used instead of optical.
- `frontend/src/components/SatelliteView.tsx` — the before/after slider now shows
  the SAR pair (not the optical one) when the execution trace says SAR was
  activated, so what the judge sees matches what the model actually analyzed.

**Verified, not just written:** built synthetic GeoTIFFs (a genuinely "cloudy"
85%-bright-uniform pair and a "clear" pair) and drove the real FastAPI app through
`TestClient` — confirmed all four decision branches fire correctly: SUFFICIENT on a
clear pair, UNSUPPORTED on a cloudy pair with no SAR, `insufficient_sar_activated`
on a cloudy pair with a complete SAR pair (and the change map is computed on the SAR
images, not the cloudy optical ones), and the "only one SAR slot filled" case
correctly falls through to UNSUPPORTED rather than pretending to activate.

---

## 2. CONFLICTING evidence state

**The gap:** named in the master brief §4B as one of four Evidence Sufficiency
Engine outcomes, never implemented through Phase 11 (PHASE11.md §5 flagged it as
deferred).

**What this is, precisely — read before demoing:** `backend/evidence/fusion.py`
cross-checks two signals SatQuery already computes independently for the same
request. It is **not** a second model and does **not** invent an independent
verification signal:
- `change_vqa`: the deterministic pixel-diff `change_map` magnitude (high/medium/
  low) vs. the free-text answer from the composite-image VQA bridge. If the
  pixel-diff says LOW and the answer asserts a specific change occurred (or the
  pixel-diff says HIGH and the answer denies any change), that's flagged as
  conflicting.
- `optical_sar`: fused `agreement_pct` vs. `optical_only_pct`/`sar_only_pct`. If
  each modality independently flags a substantial, largely non-overlapping area for
  the same target (agreement is a minority of what either sensor flagged alone),
  that's flagged as conflicting — the two sensors are pointing at different things,
  not confirming each other.

Both checks are plain arithmetic/keyword heuristics on numbers/strings SatQuery
already has. `api/query.py`'s `_run_change_vqa`/`_run_optical_sar` now call these,
add a `Verification Trace: CONFLICTING — …` line to the execution trace, and set a
new `status: "conflicting_evidence"` on the response (added to `QueryResponse` and
mirrored in `frontend/src/lib/api.ts`). The frontend (`ChatPanel.tsx`) renders this
with a distinct orange banner and styling — separate from the existing amber
"insufficient evidence" treatment, so a judge can tell the two states apart at a
glance.

**Verified:** crafted an optical/SAR pair with disjoint "water-like" regions in
opposite corners and confirmed, via a real HTTP request to `/api/query`, that the
response comes back with `status: "conflicting_evidence"` and the correct trace
line and note.

---

## 3. Real Leaflet map

**The gap:** every prior phase's README/PHASE report flagged this — "Satellite
View is currently an image/overlay viewer, not a georeferenced map," and the master
brief §4E/§7 named click-point/draw-rectangle/draw-polygon region selection as an
explicit UX target that was never built.

**What was built:** `frontend/src/components/GeoMap.tsx`, a new "Map" tab
(`MenuBar.tsx`, `page.tsx`) sitting alongside the existing Dashboard tab (which is
untouched and still the primary VQA/grounding/change/fusion workflow):
- Real Leaflet map with OpenStreetMap tiles, centered on the Assam hero-demo region.
- `leaflet-draw` toolbar: point / rectangle / polygon selection, exactly the
  interaction pattern from the master brief §7.
- A drawn selection is converted to a human-readable region description (e.g. "the
  rectangular region spanning 26.1500°N–26.3000°N, 92.8000°E–93.0000°E") and handed
  back to the page-level `pendingInsert` state, which `ChatPanel.tsx` picks up and
  inserts into the chat input on next render — so "draw a rectangle, then ask a
  question about it" actually works end-to-end.
- **Honest scope limit, stated in the UI itself, not hidden:** an uploaded image is
  only overlaid on the map when it is genuinely EPSG:4326 (plain lat/lon)
  georeferenced. Most real GeoTIFF products — including Cartosat-2S/RISAT-class
  imagery — ship in a projected CRS (typically UTM), and this component does not
  reproject. When no uploaded image qualifies, the map says so explicitly rather
  than silently placing an image in the wrong location or omitting the caveat.

**Verified:** `npm install` (added `leaflet`, `leaflet-draw`, and their `@types`
packages) and a full `next build` both pass cleanly — this is a real, compiling
build, not just code that looks plausible. (Interactive map behavior itself — tile
loading, draw events — needs a browser and wasn't exercised headlessly in this
session; the build/type-check is what was verified here.)

**Pre-existing bug fixed as a side effect:** the build surfaced a real TypeScript
error in `BenchmarksView.tsx` (`DATASETS` array had inconsistent object shapes,
`as const` inferred a union type without an optional `link` field on every
variant) that predates this phase and was never caught because the frontend had
apparently never been run through `next build` before. Fixed with an explicit type
annotation; unrelated to this phase's actual scope but blocking a clean build.

---

## 4. BigEarthNet adaptation — attempted, not completed, stated honestly

This remains, as every prior phase report has said, the single highest-priority
open item. What actually happened this session:

- `adaptation/prepare_bigearthnet.py` was **smoke-tested for real** against a
  small, clearly-synthetic manifest (5 rows, random-noise images, invented
  captions/labels — explicitly not real BigEarthNet data) and produced a correctly
  shaped training JSONL. This proves the parsing/templating code runs; it does
  **not** verify the column-name assumptions against the real PS-linked dataset,
  which was never fetched.
- `adaptation/train_lora_vqa.py` was **not run**, even on synthetic data. Two
  infrastructure constraints made this genuinely infeasible in the sandbox used to
  build this phase, not a shortcut: the dataset host is unreachable, and
  `huggingface.co` (needed for the base BLIP-VQA-base weights) is unreachable.
- `backend/models/checkpoints/vqa_lora_bigearthnet/` is still empty.
  `adapted: false` is still the correct, honest state reported by the API.

See `backend/adaptation/README.md` §"Phase 12 — what was actually verified this
session" for the full disclosure. **Do not tell a judge this has been run — it has
not.** The next actual step is someone with access to both the real dataset and a
GPU (or at least a machine that can reach huggingface.co) running the existing,
unmodified `train_lora_vqa.py`.

---

## 5. Honesty checklist (read before demoing)

- ☐ Change-task SAR activation requires uploading **both** SAR Before and SAR
  After — a single SAR image will not trigger it, by design.
- ☐ `conflicting_evidence` is a real state now — if you see it in a demo, that's
  the engine correctly flagging disagreement between two of its own signals, not a
  bug. Don't present either signal as the "real" answer when this fires.
- ☐ The Map tab's image overlay only appears for genuinely EPSG:4326-georeferenced
  uploads. If your demo GeoTIFFs are in UTM (likely, for real Cartosat-2S/RISAT
  data), the map will correctly show no overlay — that's accurate, not broken.
- ☐ `adapted: false` is still correct for every VQA/grounding response. No LoRA
  checkpoint was trained this phase.

---

## 6. What's still NOT done

- Real BigEarthNet LoRA training run (§4 above) — top priority, unchanged.
- Grounding has no adaptation pipeline of its own (unchanged from Phase 6/9/11).
- No CRS reprojection for the Map tab's image overlay (documented limitation, not a
  bug — see §3).
- Multi-model verification (master brief §13B) beyond the two-signal cross-checks
  in §2 — a larger, separate feature, not attempted here.
