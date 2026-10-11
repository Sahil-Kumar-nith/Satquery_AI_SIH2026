# SatQuery AI — Phase 9 Report

**Scope:** Evidence Sufficiency Engine — the master brief's PRIMARY HERO FEATURE.
Optical cloud obstruction → evidence insufficiency → SAR activation → answer.
**Status:** ✅ Complete for VQA and grounding. Not yet wired into change_detection/change_vqa/optical_sar (see §6).

---

## 1. What this phase actually is

Phases 1–8 built individual tools (VQA, grounding, change detection, optical+SAR fusion) that only ever run when the user's query text explicitly asks for them. Phase 9 is the first phase that makes SatQuery **decide something on its own**: if you ask a plain question against a cloudy optical image, and you also uploaded a SAR image of the same session, SatQuery now automatically switches to SAR instead of answering badly (or not at all) from a cloud-obscured photo.

This satisfies the master brief's core innovation directly:

> "The system should determine whether available evidence is sufficient for a query and activate complementary sensing/models when necessary."

---

## 2. Read this before demoing — what "SAR activation" does and does NOT mean

**It does NOT fetch a new SAR image from any satellite archive.** There is no real data feed in this prototype. "Activation" means: a SAR image you already uploaded to the same chat session is picked up and used instead of the cloudy optical one. If you didn't upload one, there is nothing to activate — SatQuery says so honestly instead of pretending.

This matches the master brief's own caution:

> "Do NOT claim that an exact co-registered SAR image is always available. Use wording such as 'best available complementary SAR observation' when appropriate."

---

## 3. How the decision works

### Step 1 — Cloud coverage estimate (`backend/evidence/cloud.py`)
A pixel is counted as "cloud-like" if it's both bright and low-saturation (close to white/grey) after the same percentile stretch used everywhere else in this codebase. This is a simple heuristic — **not** a trained cloud model, **not** the Sentinel-2 QA60 band. It will false-positive on snow, sand, salt flats, and pale rooftops. Disclosed in every response, not hidden.

### Step 2 — Sufficiency decision (`backend/evidence/sufficiency.py`)
Four possible outcomes (the four states the master brief §4B names):

| Cloud % | SAR uploaded? | Outcome | What happens |
|---|---|---|---|
| < 40% | — | **SUFFICIENT** | Answer from optical, as normal |
| ≥ 40% | Yes | **INSUFFICIENT → SAR activated** | Answer from SAR instead, trace explains the switch |
| ≥ 40%, < 75% | No | **INSUFFICIENT, no fallback** | Still answers from optical, but flags the answer as unreliable in both the trace and the response `note` |
| ≥ 75% | No | **UNSUPPORTED** | Refuses to guess — `tool_output` is `null`, response `status` is `"insufficient_evidence"`, and the note tells the user to upload a SAR image |

All thresholds (40% / 75%) are prototype policy, stated as such — not a scientifically validated cutoff.

### Step 3 — Run the tool on whichever image was selected
The underlying VQA/grounding models (Phases 5/6) are unchanged — they still only ever see one image. Phase 9 just decides *which* image to hand them.

---

## 4. What was built

### `backend/evidence/cloud.py` (new)
`estimate_cloud_coverage(optical_path)` — brightness + low-saturation heuristic, returns a percentage and an honest method disclosure.

### `backend/evidence/sufficiency.py` (new)
`decide_evidence(optical_path, sar_path)` — the four-outcome decision logic described above, returns a `SufficiencyDecision` with a ready-to-append trace.

### `backend/agent/registry.py`
`vqa`, `captioning`, and `grounding` now accept **1 or 2** images (was strictly 1). The extra slot is for an optional companion SAR image — the model still only ever runs on one image; the engine picks which.

### `backend/api/query.py`
- `_run_vqa` and `_run_grounding` now call `decide_evidence()` before running their model, select the right image path, and surface the decision in both the execution trace and the response `note`.
- New `QueryResponse.status` value: `"insufficient_evidence"` (alongside existing `"routed"` / `"blocked"`), used only for the UNSUPPORTED outcome.

### `frontend/src/app/page.tsx`
Added a one-line hint under the upload row explaining the demo setup: upload only Optical + SAR (clear Before/After) to trigger the automatic switch.

---

## 5. Example response — cloudy optical, SAR available (`POST /api/query`, task=vqa)

```json
{
  "task": "vqa",
  "status": "routed",
  "execution_trace": [
    "Query received: \"Assess flood damage in this region.\"",
    "Decision Trace: classified as Visual Question Answering (no stronger signal matched — defaulted to single-image VQA)",
    "Input validation: 2 image(s) supplied — OK",
    "Tool selected: vqa_model (BLIP-VQA-base, BigEarthNet-adapted where available) — Phase 5",
    "Evidence Trace: cloud coverage estimated at 81.4% (heuristic — brightness_saturation_heuristic (deterministic, not a trained cloud model))",
    "Decision Trace: optical evidence INSUFFICIENT (cloud coverage 81.4% ≥ 40.0% threshold)",
    "Evidence Trace: complementary SAR observation available in this session",
    "Tool Execution: SAR activation — switching analysis to the uploaded SAR image",
    "Tool Execution: running VQA specialist model…",
    "Evidence Trace: model answer generated (740 ms)",
    "Evidence Trace: confidence 0.58 (MEDIUM) — Model showed moderate token-level uncertainty...",
    "Evidence Fusion: n/a — single-model, single-image task; ..."
  ],
  "tool_output": {
    "answer": "flooded area",
    "confidence": 0.58,
    "confidence_level": "medium"
  },
  "note": "Optical evidence was insufficient (cloud coverage ~81.4%) — this answer was generated from the SAR image instead. VQA answer + confidence come from a live model call..."
}
```

## 6. What's still NOT done (be honest about this in a demo)

- **No map/spatial evidence overlay yet.** The trace is text-only. Phase 10 (map visualization) is where "SAR evidence highlighted on the map" actually happens.
- **Change detection / change-VQA / optical_sar fusion do not use this engine yet.** They still run only when explicitly requested by query keywords — cloud-triggered switching only applies to VQA and grounding right now. Extending it to the other tasks is a reasonable Phase 10+ addition, not done here.
- **CONFLICTING state (cross-checking two models) is not implemented.** Only SUFFICIENT / INSUFFICIENT(+SAR) / INSUFFICIENT(no fallback) / UNSUPPORTED exist.
- **Known frontend gap:** the chat panel currently sends *every* image uploaded to the page with every query, not just the relevant ones. For the demo, Before/After must be cleared before asking a VQA/grounding question, or `validate_inputs` will reject the request for having too many images. Documented as a required workaround in the user manual; a proper fix (only send task-relevant images) is Phase 10 scope.

---

## 7. Honesty checklist (read before demoing)

- ☐ Say "cloud coverage heuristic," never "official cloud mask."
- ☐ Say "SAR image already uploaded this session," never "fetched" or "acquired" a new SAR observation.
- ☐ If `status == "insufficient_evidence"`, that's a feature working correctly (the UNSUPPORTED state), not a bug — don't apologize for it in a demo, explain it.
- ☐ `insufficient_no_fallback` answers are explicitly marked unreliable in the trace — don't quote that answer as if it were a confident result.

---

## 8. Next phase

**Phase 10 — Map visualization + spatial evidence overlay.** This is where the grounding boxes and change regions actually get drawn on a map instead of returned as raw pixel coordinates, and where the Assam monsoon hero demo becomes visually complete per the master brief's §5.
