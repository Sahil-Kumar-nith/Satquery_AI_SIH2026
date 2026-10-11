# SatQuery AI — Phase 5 Report

**Scope:** Single-image VQA specialist model integration + remote-sensing domain adaptation pipeline.
**Status:** ✅ Complete — model is live and callable end-to-end; adaptation pipeline is built and runnable but not yet executed (no adapter checked in).

---

## 1. PS requirements this phase satisfies

| PS requirement | How Phase 5 satisfies it |
|---|---|
| "Single-image VQA shall be mandatory." | `POST /api/query` now runs a real model for `task: "vqa"` and returns a generated answer — not a routing stub. |
| "At least one visual or vision-language component must be fine-tuned or otherwise adapted using BigEarthNet.txt or any open source training data." | `backend/adaptation/` — a real, runnable LoRA fine-tuning pipeline against BigEarthNet.txt-derived data. See §4 for its current honest status. |
| "...combine textual and spatial outputs, estimate confidence..." | VQA responses include a confidence score + HIGH/MEDIUM/INSUFFICIENT bucket (no spatial output yet — that starts with grounding in Phase 6). |
| "...provide an auditable execution summary containing the selected task, model/tool names, and key parameters." | Execution trace now includes real timing, the exact model ID used, and whether adaptation was applied — all observable, no hidden reasoning exposed. |

---

## 2. What was built

### Model: `backend/models/vqa.py`
- **Model:** `Salesforce/blip-vqa-base` (BSD-3, ~385M params, `transformers`-only, CPU-viable).
- **Why this one:** verified to actually take (image, question) → text answer; low integration complexity; no GPU dependency assumed. A 7B-parameter RS-native VLM (e.g. GeoChat) was considered and rejected for this phase — unrealistic to assume adequate GPU access within a 4-day hackathon budget.
- **Confidence:** mean per-token generation probability from the model's own decode step. This is a **decoding-confidence proxy**, not a calibrated accuracy number — never present it as validated benchmark performance.
- **Adaptation-aware:** if a LoRA adapter exists at `backend/models/checkpoints/vqa_lora_bigearthnet/`, it's loaded automatically and `adapted: true` is reported. If not, the base model runs and the response says so honestly (`adapted: false`).

### Adaptation pipeline: `backend/adaptation/`
- `prepare_bigearthnet.py` — converts a BigEarthNet.txt-style manifest (captions or land-cover labels) into templated VQA training triplets.
- `train_lora_vqa.py` — LoRA fine-tuning of `blip-vqa-base` on those triplets using `peft` + HF `Trainer`.
- **Not yet run.** This is disclosed, not hidden — see `backend/adaptation/README.md` for exactly what needs to happen (verify the real PS-linked dataset's manifest format, decide compute, then run it) before the demo.

### Supporting change: `backend/preprocessing/raster_to_rgb.py`
Shared band-selection/percentile-stretch logic so every future specialist model (grounding, change detection, optical+SAR) can turn any validated raster — 1-band SAR, multi-band optical GeoTIFF, PNG/JPEG — into a plain RGB image without reimplementing this per model.

### API change: `backend/api/query.py`
`task: "vqa"` + valid inputs now triggers a real model call instead of stopping at `"pending"`. Every other task (captioning, grounding, change, optical+SAR) is untouched — still routing-only, `tool_output: null`, honestly labeled as pending its own phase.

### Frontend change
`ChatPanel.tsx` renders the answer, a confidence badge (HIGH/MEDIUM/INSUFFICIENT), an "not yet RS-adapted" flag when `adapted: false`, and inference latency.

---

## 3. Example response (`POST /api/query`, task=vqa)

```json
{
  "task": "vqa",
  "status": "routed",
  "selected_tool": "vqa_model (BLIP-VQA-base, BigEarthNet-adapted where available) — Phase 5",
  "tool_output": {
    "answer": "water is visible in the lower left region",
    "confidence": 0.62,
    "confidence_level": "medium",
    "confidence_note": "Model showed moderate token-level uncertainty — treat as indicative, not definitive.",
    "model_id": "Salesforce/blip-vqa-base (base, not yet adapted)",
    "adapted": false,
    "latency_ms": 840
  },
  "execution_trace": [
    "Query received: \"Is there water visible in this image?\"",
    "Decision Trace: classified as Visual Question Answering (no stronger signal matched — defaulted to single-image VQA)",
    "Input validation: 1 image(s) supplied — OK",
    "Tool selected: vqa_model (BLIP-VQA-base, BigEarthNet-adapted where available) — Phase 5",
    "Tool Execution: running VQA specialist model…",
    "Evidence Trace: model answer generated (840 ms)",
    "Evidence Trace: confidence 0.62 (MEDIUM) — Model showed moderate token-level uncertainty — treat as indicative, not definitive.",
    "Evidence Fusion: n/a — single-model, single-image task; fusion applies once optical+SAR (Phase 8) or multi-model verification is in play"
  ]
}
```

---

## 4. Honesty checklist (read before demoing)

- ☐ `adapted` field checked in every response before claiming "remote-sensing adapted" to a judge.
- ☐ Confidence described as a decoding-confidence proxy, not benchmark accuracy, in any pitch/slide.
- ☐ Every other task still correctly returns `tool_output: null` — don't let a demo script imply otherwise.
- ☐ If the LoRA adapter was actually trained, note what dataset/subset/step-count was used — don't claim a scale of adaptation that didn't happen.

---

## 5. Setup delta from Phase 4

```bash
cd backend
pip install -r requirements.txt   # now includes torch, transformers, pillow, peft
uvicorn main:app --reload --port 8000
```
First VQA call per process will be slower (model download + load); subsequent calls reuse the in-memory singleton.

To run the (optional, not-yet-executed) adaptation step, see `backend/adaptation/README.md`.

---

## 6. Next phase

**Phase 6 — text-guided region grounding** (the second mandatory single-image task, chosen over captioning for its stronger map/demo fit — grounding produces the bounding box the PS explicitly calls for: *"Highlight the water body referred to in the query."*). Candidate model: `google/owlvit-base-patch32` (open-vocabulary detector, Apache-2.0, `transformers`-only — no new dependency footprint beyond what Phase 5 already added).
