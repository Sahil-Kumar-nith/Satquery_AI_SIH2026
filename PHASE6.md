# SatQuery AI — Phase 6 Report

**Scope:** Second mandatory single-image task — text-guided region grounding.
**Status:** ✅ Complete — model is live and callable end-to-end. No RS adaptation pipeline for this model yet (disclosed, not hidden).

---

## 1. PS requirement this phase satisfies

> "Each solution must additionally implement either captioning/scene description or text-guided region grounding."
> Representative query: *"Highlight the water body referred to in the query."*

**Decision: grounding over captioning.** Grounding produces a bounding box — spatial evidence the PS explicitly asks for and the strongest fit for the map-based UI planned in Phase 10. Captioning would satisfy the letter of the requirement with less integration work, but grounding scores better on demo quality and evidence-groundedness (both explicit evaluation criteria). If Phase 7/8 run over budget, captioning remains available as a cheap fallback — a captioning model wasn't built this phase, so pick that up only if grounding needs to be cut, not in addition to it.

---

## 2. What was built

### Model: `backend/models/grounding.py`
- **Model:** `google/owlvit-base-patch32` — Apache-2.0, `transformers`-only, no new dependency beyond what Phase 5 already installed.
- **Verified capability:** open-vocabulary object detection — image + free-text phrase → bounding box + match score. This is literally what "text-guided region grounding" requires; no detection-head retraining needed to add new phrase categories.
- **Not remote-sensing-native.** Trained on natural/web imagery. Expect solid performance on generic categories ("water", "building", "vehicle") and weaker performance on RS-specific phrasing ("built-up area", "paddy field") until adapted — same caveat as Phase 5's VQA model, not yet addressed here (see §5).
- **Referring-phrase extraction:** rule-based (`extract_referring_phrase`), strips known question scaffolding ("where is", "highlight", "locate", etc.) to turn a natural-language question into a clean noun phrase for the detector — deterministic, consistent with the router's existing approach, no LLM call.
- **Confidence:** the detector's own softmax match score for the best box, bucketed HIGH/MEDIUM/INSUFFICIENT. This is a **detector-confidence proxy**, not a validated grounding-accuracy benchmark figure.

### API change: `backend/api/query.py`
- Refactored the single growing if/elif into a `_TASK_HANDLERS` dispatch table (`TaskType -> handler function`) so Phase 7/8 can be added as isolated functions instead of one unreadable branch.
- `task: "grounding"` + valid inputs now calls the model and returns a real box + score. `change_detection` / `change_vqa` / `optical_sar` are untouched — still routing-only.

### Frontend change
`ChatPanel.tsx` renders the extracted phrase, the bounding box (pixel coordinates — explicitly labeled as "map overlay pending Phase 10"), confidence badge, and an "not yet RS-adapted" flag, following the same visual pattern as the Phase 5 VQA card.

---

## 3. Example response (`POST /api/query`, task=grounding)

```json
{
  "task": "grounding",
  "status": "routed",
  "selected_tool": "grounding_model (OWL-ViT open-vocabulary detector) — Phase 6",
  "tool_output": {
    "phrase": "water body",
    "found": true,
    "box": { "x_min": 112.4, "y_min": 340.1, "x_max": 289.7, "y_max": 501.9 },
    "score": 0.34,
    "confidence_level": "high",
    "confidence_note": "Detector was strongly confident this region matches the phrase.",
    "image_width": 1024,
    "image_height": 1024,
    "model_id": "google/owlvit-base-patch32 (base, not RS-adapted)",
    "adapted": false,
    "latency_ms": 610
  },
  "execution_trace": [
    "Query received: \"Where is the water body?\"",
    "Decision Trace: classified as Text-Guided Region Grounding (matched region-grounding keywords)",
    "Input validation: 1 image(s) supplied — OK",
    "Tool selected: grounding_model (OWL-ViT open-vocabulary detector) — Phase 6",
    "Decision Trace: extracted referring phrase \"water body\" from query",
    "Tool Execution: running grounding specialist model…",
    "Evidence Trace: region located (610 ms)",
    "Evidence Trace: confidence 0.34 (HIGH) — Detector was strongly confident this region matches the phrase.",
    "Evidence Fusion: n/a — single-model, single-image task; fusion applies once optical+SAR (Phase 8) or multi-model verification is in play"
  ]
}
```

If no matching region is found, `found: false`, `box: null`, and the trace/note say so explicitly — never a fabricated box.

---

## 4. Confidence threshold note

OWL-ViT match scores run numerically lower than VQA's token-decoding probabilities by nature of open-vocabulary detection, so the HIGH/MEDIUM/INSUFFICIENT buckets use different absolute thresholds (0.30 / 0.12) than VQA's (0.75 / 0.40). Both are disclosed as prototype policy, not scientifically validated cutoffs, per the master brief's confidence guidance.

---

## 5. Honesty checklist (read before demoing)

- ☐ `adapted` is `false` for every grounding response right now — no RS adaptation pipeline exists for this model yet. Don't imply otherwise.
- ☐ Box coordinates are pixel coordinates on the uploaded image only — no georeferencing, no map overlay. Say "detected region" in a demo, not "mapped location."
- ☐ Score described as detector match confidence, not grounding accuracy, in any pitch/slide.
- ☐ `change_detection`, `change_vqa`, `optical_sar` still correctly return `tool_output: null`.

---

## 6. Next phase

**Phase 7 — bi-temporal change detection / change-VQA.** This is the other mandatory multi-image capability (alongside optical+SAR in Phase 8) and currently has zero model behind it — highest-priority remaining gap. Needs a before/after image pair; candidate approach and model selection to be scoped next.
