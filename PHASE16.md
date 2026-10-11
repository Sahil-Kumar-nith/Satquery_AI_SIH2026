# PHASE 16 — Follow-up chain hops 2 & 3, and a hop-1 bug fix

Scope for this phase: the one item from `LEFT.md` that was actually
buildable without external network access in this sandbox — extending
`agent/followup.py`'s Phase 15 chain. Everything else on `LEFT.md`
(real RS adaptation training, real benchmark evaluation, a genuine
third trained model for verification) still needs a machine with real
dataset/model-hub access — see the handoff prompt for that.

## 1. Bug fix: hop 1 ("Where exactly?") was frequently unreachable

Phase 15 wired `is_followup_query()`/`infer_target_phrase()` **inside**
`_run_grounding()` — but `agent/router.py`'s `classify_query()` only
recognises `"where is"` (not bare `"where exactly?"`, `"show it"`, etc.)
as grounding-intent. A literal `"Where exactly?"` follow-up would
default to VQA classification and never reach `_run_grounding()` at
all, so the Phase 15 phrase-inference code was often dead code in
practice.

**Fix:** `agent/followup.py` gained `followup_intent()`, a richer
classifier checked in `api/query.py`'s `run_query()` **before**
`classify_query()`. When a follow-up intent is detected, the task is
set directly (`grounding` / `area_estimate` / `sar_cross_check`),
bypassing keyword classification entirely. This makes all three hops
reliably reachable.

## 2. Hop 2 — "How much area?" (`evidence/area.py`, new)

Pure geometry over the **previous turn's grounding box** — not a new
model call. Converts the box's pixel footprint to a ground-distance
footprint using the source image's own GeoTIFF bounds/width/height,
then reports km².

**Honesty contract (read before demoing):** a real number is only ever
produced when the source image has a projected/metric CRS (e.g. a UTM
zone). For PNG/JPEG benchmark images (no CRS at all) or GeoTIFFs in a
geographic/degree-based CRS (e.g. plain EPSG:4326), this deliberately
reports "area unavailable" with a clear reason rather than silently
treating degrees as metres, which would be badly wrong depending on
latitude. This is a rectangular bounding-box footprint, not a
segmentation of the actual object — treat it as an order-of-magnitude
estimate.

## 3. Hop 3 — "Was this visible in SAR?" (`_run_sar_check_followup` in `api/query.py`)

Re-runs the **same** OWL-ViT grounding model against whichever SAR
image was supplied this session, using a phrase inferred from the
previous turn's answer text. This is an independent detector call on
different pixels — explicitly **not** a claim that optical and SAR were
jointly fused (that's the dedicated Phase 8 `optical_sar` task).

## 4. New task types (`agent/registry.py`)

`area_estimate` and `sar_cross_check` were added to `TaskType` and
`TASK_REGISTRY`, both explicitly marked `NOT PS-mandatory` in their
`ps_requirement` field — they only exist to support the follow-up
chain and are only ever reached via `followup_intent()`, never via
`agent/router.py`'s keyword classifier.

## 5. Plumbing changes

- `QueryRequest` gained `previous_box` and `previous_image_id` (Phase
  15 already had `previous_query`/`previous_answer`).
- `GroundingOutput` gained `image_id` — the specific uploaded image
  (optical or SAR) a box was actually computed on, since the Evidence
  Sufficiency Engine may pick either. Needed so hop 2 knows which
  image's georeferencing to use.
- Frontend (`lib/api.ts`, `ChatPanel.tsx`): `FollowupContext` now
  carries `previousBox`/`previousImageId`, pulled from the last
  grounding result via a new `extractPreviousGrounding()` helper.
  `ResultBody` gained a render branch for `AreaEstimateOutput`.

## What this phase deliberately did NOT attempt

Same network constraint as Phase 15 — this sandbox cannot reach
`huggingface.co`, `arxiv.org`, or any dataset/model host:

- Did NOT run real RS adaptation training (VQA or grounding).
- Did NOT run any real public-benchmark evaluation.
- Did NOT add a genuine second trained model to
  `evidence/verification.py` — that would need downloading a
  pretrained change-detection model from a hub this sandbox can't
  reach. (Correction: an earlier chat turn said this needed "no
  internet" — that was wrong. It does.)

See the separate handoff prompt (given to the user alongside this
zip) for exactly what to run, in what order, on a machine with real
internet access.

## What was NOT verified end-to-end in this session

- The frontend TypeScript changes were checked for brace/paren balance
  and structural consistency only — `node_modules` isn't installed in
  this sandbox, so `tsc`/`next build` was not run. Run `npm install &&
  npm run build` (or `tsc --noEmit`) before trusting this compiles
  clean.
- The backend follow-up intent classifier and area-estimation math
  were unit-tested directly in this session (see inline examples in
  `agent/followup.py`'s and `evidence/area.py`'s docstrings). The full
  `/api/query` request/response cycle for all three hops was **not**
  exercised against a running FastAPI server + real uploaded images in
  this session — do that before a live demo.
