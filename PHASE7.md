# SatQuery AI — Phase 7 Report

**Scope:** Mandatory multi-image change analysis — bi-temporal change detection AND change-based VQA.
**Status:** ✅ Complete — both task types are live and callable end-to-end. Deterministic heuristic, no trained change model (disclosed, not hidden — see §2).

---

## 1. PS requirement this phase satisfies

> "Multi-image change analysis: Change description or change-based visual question answering from a bi-temporal image pair shall be mandatory. A spatial change map may also be generated where reference masks are available."

Both `change_detection` (description + change map) and `change_vqa` (free-text answer to a change question) are implemented, since the router already classifies queries into both buckets (Phase 4) and both are cheap to support from the same underlying change map.

---

## 2. Why deterministic, not a trained model — read this before demoing

There is no pip-installable, genuinely pretrained bi-temporal RS change-detection network that fits "prefer existing pretrained models" + "avoid unnecessary infrastructure" + a 4-day build. Toolkits that exist (open-cd, torchange, etc.) need dataset-specific checkpoints and non-trivial setup — not a hackathon-week integration. The master brief explicitly sanctions this fallback for exactly this situation: *"You don't need a perfect model. You need working end-to-end output"* and names *"simple fusion/classifier"* as an acceptable MVP approach.

**What's actually running:**
- **Change map:** registered-pixel absolute difference on percentile-stretched grayscale, thresholded at `mean + 1.5·std`, summarized into a changed-pixel percentage and the bounding box of the largest changed region. This is classical CV, not a learned detector — it reacts to illumination/seasonal/sensor differences as well as real change.
- **Change description:** rule-based sentence generation from the change stats (percentage + level bucket). No model call.
- **Change-VQA:** the before/after pair is stitched into one side-by-side composite image and fed into the **Phase 5 BLIP-VQA model** along with the user's question. This is a composite-image VQA heuristic, not a purpose-built CDVQA model — CDVQA is named in the PS as an *evaluation benchmark* for this capability, not a mandated architecture, so this is a legitimate (if honestly-labeled) way to satisfy the requirement in 4 days.

Both paths run on real pixels from the actual uploaded images — nothing here is a canned/demo constant.

---

## 3. What was built

### `backend/models/change.py`
- `compute_change_map(before_path, after_path)` — deterministic diff + threshold + largest-region box. Resizes mismatched pixel dimensions with a disclosed caveat (this is NOT geometric co-registration).
- `describe_change(change, query)` — rule-based text description.
- `run_change_vqa(before_path, after_path, query)` — builds the before|after composite, calls the Phase 5 VQA singleton (no new model weights loaded), returns the answer alongside the independent change-map stats so judges can sanity-check one against the other.

### `backend/preprocessing/raster_to_rgb.py`
- Exposed the existing percentile-stretch helper as a public `to_uint8` (was private `_to_uint8`) so Phase 7/8 can reuse it for grayscale diffing without duplicating the stretch logic.

### `backend/api/query.py`
- Added `_run_change_detection` and `_run_change_vqa` handlers, registered in `_TASK_HANDLERS`.
- Added a shared `_before_after_paths()` helper that resolves the `before`/`after` roles already enforced by Phase 4's `validate_inputs()` — no new validation logic needed here, it was already correct.

### `backend/agent/registry.py`
- `change_detection` / `change_vqa` `tool` strings updated from "not yet integrated" to the real tool names now that models exist.

---

## 4. Example response — change detection (`POST /api/query`, task=change_detection)

```json
{
  "task": "change_detection",
  "status": "routed",
  "tool_output": {
    "description": "HIGH change detected: 11.3% of pixels differ between the before and after images, concentrated in a concentrated region.",
    "change_map": {
      "changed_pixel_pct": 11.3,
      "change_level": "high",
      "change_note": "11.3% of pixels flagged as changed by a statistical threshold (>1.5 std-dev above mean intensity difference). This is a classical pixel-difference heuristic, not a learned change-detection model...",
      "largest_change_box": { "x_min": 210.0, "y_min": 88.0, "x_max": 640.0, "y_max": 512.0 },
      "width": 1024, "height": 1024,
      "method": "pixel_diff_zscore_threshold (deterministic, not a trained model)"
    }
  }
}
```

## 5. Example response — change-VQA (`POST /api/query`, task=change_vqa)

```json
{
  "task": "change_vqa",
  "status": "routed",
  "tool_output": {
    "answer": "yes",
    "vqa_confidence": 0.61,
    "vqa_confidence_level": "medium",
    "model_id": "Salesforce/blip-vqa-base (base, not yet adapted) (composite before|after image — no dedicated CDVQA model)",
    "change_map": { "changed_pixel_pct": 11.3, "change_level": "high", "...": "..." }
  }
}
```

---

## 6. Honesty checklist (read before demoing)

- ☐ Say "pixel-difference heuristic" or "prototype change policy" — never "AI-detected change" unqualified.
- ☐ Change-VQA answer is a general VLM's guess on a stitched image — say "prototype composite-image VQA," never imply a dedicated change-VQA/CDVQA model.
- ☐ If before/after images have different pixel dimensions, the response says so — don't hide that caveat in a demo.
- ☐ `vqa_confidence` is BLIP's decoding-confidence proxy (same caveat as Phase 5), not a validated accuracy number.

---

## 7. Next phase

**Phase 8 — optical+SAR cross-modal analysis.** Compiled in this same delivery — see `PHASE8.md`. Phase 9 (Evidence Sufficiency Engine — the actual "cloud → SAR activation" decision logic) is next after that; Phase 7 and 8 build the specialist tools it will call, not the decision logic itself.
