# SatQuery AI

**Smart India Hackathon — Problem Statement 26167**
*SatQuery AI — An Interactive Vision-Language Assistant for Multimodal Remote Sensing Image Analysis through Text Queries*

**Organization:** Indian Space Research Organisation (ISRO), Department of Space
**Category:** Software · **Theme:** Space Technology

---

## 1. What this is

SatQuery AI is an agentic vision-language assistant for analysing single and paired remote-sensing images (optical/multispectral and SAR) through natural-language queries. Instead of applying one generic vision-language model to every question, the system:

1. classifies what the query is actually asking for,
2. checks whether the supplied imagery is sufficient and compatible with that task,
3. selects the appropriate specialist model/tool from a predefined registry,
4. executes it — VQA, grounding, change detection, change-VQA, and optical+SAR fusion are all live as of Phase 8 — and fuses the results, and
5. returns an evidence-grounded answer with a confidence level and an **auditable execution trace** — never a hidden chain-of-thought.

### Core innovation — Adaptive Evidence Acquisition
The system determines whether currently available imagery is sufficient to answer a query. When it isn't — most demonstrably when optical imagery is cloud-obstructed — the system activates a complementary sensing modality (SAR) rather than answering unreliably or refusing outright.

**Hero feature:** Optical cloud obstruction → evidence insufficiency detected → SAR activation → multimodal evidence fusion → answer + spatial evidence + confidence + execution trace.

**Status: implemented for VQA and grounding (Phase 9), and generalized to change tasks (Phase 12).**
Phase 11 extended the sufficiency check to change_detection/change_vqa but, at the time, could only
ever detect the evidence gap, not fail over to SAR (no SAR-before/SAR-after upload slot existed
yet). Phase 12 closed that: SAR Before/After upload slots now exist, and a cloudy Before/After pair
genuinely fails over to a SAR Before/After pair when both are supplied — see `PHASE12.md`.

---

## 2. Mandatory PS requirements

| # | Requirement | Status |
|---|---|---|
| 1 | Single-image VQA | **Working** — Phase 5: BLIP-VQA-base, live model call; Phase 9 adds cloud→SAR evidence switching |
| 2 | One additional single-image task (captioning or grounding) | **Working** — Phase 6: text-guided region grounding, OWL-ViT live model call; Phase 9 evidence switching too |
| 3 | Bi-temporal change analysis | **Working** — Phase 7: change detection + change-VQA, deterministic pixel-diff heuristic; Phase 12 adds real cloud→SAR-pair fallback (Phase 11 only detected the gap) |
| 4 | Optical + SAR cross-modal analysis | **Working** — Phase 8: percentile-threshold agreement fusion; Phase 12 adds a CONFLICTING check when the two modalities substantially disagree |
| 5 | Agentic model/tool orchestration | **Working** — rule-based query classification + input validation + tool selection + execution |
| 6 | GeoTIFF/TIFF input support | **Working** — Rasterio-based validation |
| 7 | Input validation | **Working** |
| 8 | Model/tool selection | **Working** (registry-driven) |
| 9 | Evidence-grounded responses | **Working** — answers/boxes/maps come from real model calls or deterministic heuristics, not templates |
| 10 | Confidence information | **Working** — decoding-confidence / detector-score / agreement-% proxies, see §5 caveat |
| 11 | Observable execution trace | **Working** |
| 12 | Interactive GUI/web application | **Working** — Satellite View panel (Phase 10) plus a real Leaflet map tab with click-point/draw-rectangle/draw-polygon selection (Phase 12) |

Remote-sensing adaptation (fine-tuning/domain adaptation using BigEarthNet.txt or another permitted open-source source) has runnable scripts as of Phase 5 (`backend/adaptation/`), but **no adapter has actually been trained and checked in yet** — the shipped VQA model is honestly reporting `adapted: false` until someone runs the pipeline. The grounding model (Phase 6) has no adaptation pipeline of its own yet either — see §9. **This remains the single highest-priority open item** — everything else in this table is functionally complete.

---

## 3. Architecture

```
                    ┌───────────────────────┐
                    │   NEXT.JS FRONTEND    │
                    │  Map | Chat | Upload  │
                    └───────────┬───────────┘
                                │ REST (JSON)
                                ▼
                    ┌───────────────────────┐
                    │  FASTAPI BACKEND      │
                    │  /api/health          │
                    │  /api/upload          │
                    │  /api/query           │
                    └───────────┬───────────┘
                                │
                ┌───────────────┼────────────────┐
                ▼                                 ▼
        Input Validator                    Agent Router
        (preprocessing/geotiff.py)         (agent/router.py)
        - Rasterio format/CRS/band check   - Rule-based task classification
        - Powers /api/upload                - Input compatibility check
                                             - Tool selection (agent/registry.py)
                                                     │
                                                     ▼
                                          ┌─────────────────────┐
                                          │ SPECIALIST MODELS   │
                                          │ VQA ✅ (Phase 5)     │
                                          │ Grounding ✅ (Phase 6)│
                                          │ Change ⏳ (P7)       │
                                          │ Optical+SAR ⏳ (P8)  │
                                          └─────────────────────┘
                                                     │
                                                     ▼
                                          Confidence (per-model, Phase 5+)
                                          Multi-source Evidence Fusion
                                          (Phase 9, not yet built)
                                                     │
                                                     ▼
                                          Answer + Execution Trace
```

### Tech stack
- **Frontend:** Next.js + TypeScript + Tailwind
- **Backend:** FastAPI + Python
- **AI:** PyTorch + Hugging Face `transformers`. Two specialist models live: `Salesforce/blip-vqa-base` (VQA, + optional LoRA adapter via `peft`) and `google/owlvit-base-patch32` (grounding). Change detection and optical+SAR fusion land in Phases 7-8.
- **Geospatial:** GDAL / Rasterio / GeoPandas
- **Map (planned):** Leaflet or OpenLayers

---

## 4. Phase progress

| Phase | Scope | Status |
|---|---|---|
| 1 | Backend bootstrap — FastAPI app, CORS, `GET /api/health` | ✅ Done |
| 2 | Frontend shell — Next.js dashboard layout, system status indicator | ✅ Done |
| 3 | Image upload & validation — `POST /api/upload`, Rasterio metadata extraction, drag/drop UI | ✅ Done |
| 4 | Query/Agent Router — `POST /api/query`, rule-based task classification, input compatibility checks, tool registry, execution trace | ✅ Done |
| 5 | Single-image VQA specialist model integration (BLIP-VQA-base) + remote-sensing adaptation pipeline (BigEarthNet LoRA, scripts ready) | ✅ Done — model live; adapter training is still a follow-up run, not code (see §2) |
| 6 | Second single-image task — text-guided region grounding (OWL-ViT), chosen over captioning for map/demo fit | ✅ Done — live model call, pixel bounding boxes; map overlay landed in Phase 10 |
| 7 | Bi-temporal change detection / change-VQA specialist model | ✅ Done — deterministic pixel-diff change map + rule-based description (`models/change.py`) |
| 8 | Optical + SAR cross-modal fusion specialist model | ✅ Done — percentile-threshold agreement fusion (`models/optical_sar.py`) |
| 9 | Adaptive Evidence Acquisition engine — cloud detection, evidence sufficiency scoring, SAR interception (hero feature) | ✅ Done for VQA + grounding (see PHASE9.md) |
| 10 | Map visualization — Satellite View panel: grounding overlay, before/after slider, optical+SAR agreement bars; task-aware image routing fix | ✅ Done for the P0 subset (see PHASE10.md) — no real georeferenced map (Leaflet/OpenLayers) yet |
| 10.5 | Console UI shell — top menu bar, 3 switchable console themes, Model Registry / Benchmarks / Report tabs | ✅ Done (UI-only pass) |
| 11 | Generalize the Evidence Sufficiency Engine to change_detection/change_vqa (cloud-coverage check on Before/After pairs) | ✅ Done — see PHASE11.md for the real limitation (no SAR fallback for change tasks yet) |
| 12 | SAR-before/SAR-after upload slots + real SAR fallback for change tasks, CONFLICTING evidence state, real Leaflet map with click/draw region selection, attempted BigEarthNet adaptation run | ✅ Done for the first three — see PHASE12.md. Adaptation run attempted but not completed (dataset/HF-hub host unreachable in the build sandbox); `adapted: false` remains correct |

**Current state: Phases 1–12 functionally complete for a hackathon MVP.** All six task types
(VQA, captioning-substitute/grounding, change detection, change-VQA, optical+SAR) run real models
or deterministic heuristics — nothing returns a fabricated answer. Cloud-driven evidence
switching now works for VQA/grounding AND change tasks (given a complete SAR before/after pair).
A CONFLICTING evidence state exists and is verified to fire on genuinely disagreeing signals. A
real georeferenced map with region-selection exists alongside the original Satellite View. The one
item every phase report has consistently flagged as still open:

**Honesty note, unchanged across every phase:** none of the live models are remote-sensing-native
until someone actually runs an adaptation pipeline — check the `adapted` field in every response
before claiming domain adaptation to a judge. Only VQA has an adaptation pipeline built at all
(`backend/adaptation/`), and it has not been run — see §9.

---

## 5. API reference (implemented)

### `GET /api/health`
Liveness check. Returns `{status, service, timestamp}`.

### `POST /api/upload`
Multipart form: `file`, `modality` (`optical`|`sar`), `role` (`single`|`before`|`after`).
Validates with Rasterio, stores the file, returns structured metadata (`image_id`, dimensions, band count, CRS, bounds, `is_geospatial`, etc.). In-memory registry — resets on backend restart.

### `GET /api/upload/{image_id}`
Look up previously uploaded image metadata.

### `POST /api/query`
Body: `{ query: string, image_ids: string[] }`.
Classifies the task (`vqa` / `captioning` / `grounding` / `change_detection` / `change_vqa` / `optical_sar`), validates the supplied images against that task's requirements, selects a tool from the registry, and returns a `routed` or `blocked` status with a step-by-step execution trace.

For `task: "vqa"` with valid inputs, `tool_output` is populated with a real result:
```json
{
  "answer": "water is visible in the lower left region",
  "confidence": 0.62,
  "confidence_level": "medium",
  "confidence_note": "Model showed moderate token-level uncertainty — treat as indicative, not definitive.",
  "model_id": "Salesforce/blip-vqa-base (base, not yet adapted)",
  "adapted": false,
  "latency_ms": 840
}
```
For `task: "grounding"` with valid inputs, `tool_output` looks like:
```json
{
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
}
```
`box` coordinates are pixel coordinates on the originally uploaded image — no CRS/geo-transform applied yet, and no map overlay (Phase 10). `confidence`/`score` in both cases are proxies from the model's own output (decoding probability for VQA, detector match score for grounding) — **not** calibrated benchmark accuracy numbers. Do not present them as validated performance. `change_detection`/`change_vqa`/`optical_sar` still return `tool_output: null` — see Phase 7+.

---

## 6. Project structure

```
satquery/
├── backend/
│   ├── main.py                    # FastAPI composition root
│   ├── requirements.txt
│   ├── api/
│   │   ├── health.py              # GET /api/health
│   │   ├── upload.py              # POST /api/upload, GET /api/upload/{id}
│   │   └── query.py               # POST /api/query
│   ├── agent/
│   │   ├── registry.py            # Task specs: input requirements + tool names
│   │   └── router.py               # classify_query() + validate_inputs()
│   ├── preprocessing/
│   │   ├── geotiff.py             # Rasterio-based format/CRS/band validation
│   │   └── raster_to_rgb.py       # Any validated raster -> PIL RGB for model input
│   ├── models/
│   │   ├── vqa.py                 # Phase 5: BLIP-VQA-base + optional LoRA adapter
│   │   ├── grounding.py           # Phase 6: OWL-ViT text-guided region grounding
│   │   └── checkpoints/
│   │       └── vqa_lora_bigearthnet/  # empty until adaptation/train_lora_vqa.py is run
│   ├── adaptation/                # Phase 5: PS-mandatory RS domain adaptation pipeline (VQA only so far)
│   │   ├── prepare_bigearthnet.py # BigEarthNet.txt manifest -> VQA training triplets
│   │   ├── train_lora_vqa.py      # LoRA fine-tune of BLIP-VQA-base
│   │   └── README.md              # status + what to verify before running for real
│   ├── evidence/                  # Evidence fusion/confidence — empty, Phase 9
│   ├── data/
│   │   ├── uploads/                # Runtime-created, uploaded files
│   │   └── demo/                  # Demo scenario assets
│   └── utils/
│
├── frontend/
│   └── src/
│       ├── app/page.tsx           # Main dashboard: map + chat + upload toolbar
│       ├── components/
│       │   ├── SystemStatus.tsx
│       │   ├── UploadSlot.tsx     # Drag/drop upload wired to /api/upload
│       │   └── ChatPanel.tsx      # Chat UI wired to /api/query
│       └── lib/api.ts             # Typed API client
│
└── README.md
```

---

## 7. Running locally

```bash
# Backend
cd satquery/backend
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# Frontend (separate terminal)
cd satquery/frontend
npm install
npm run dev
```

Open `http://localhost:3000`. Upload an Optical, SAR, Before, or After image (`.tif`/`.tiff` for full geospatial validation, or `.png`/`.jpg` for benchmark datasets), then ask a question in the chat panel. The response shows the classified task, an `OK`/`FAILED` input check, and the execution trace.

---

## 8. Explicit scope boundaries (read before extending)

- **In-memory registries only.** `IMAGE_REGISTRY` in `api/upload.py` is per-process and clears on restart. Fine for a hackathon demo; do not treat it as persistent storage.
- **Only VQA and grounding have specialist models.** `agent/registry.py` still lists placeholder tool names for change/optical+SAR — `POST /api/query` deliberately returns `tool_output: null` for those. Do not fabricate answers ahead of Phases 7-8.
- **Neither live model is RS-adapted by default.** VQA's `adapted` flips to `true` only once `backend/adaptation/train_lora_vqa.py` has actually been run against a verified BigEarthNet.txt manifest (see `backend/adaptation/README.md`). Grounding has no adaptation pipeline yet at all — its `adapted` field is always `false` until one is built.
- **Confidence/score fields are model-native proxies**, not validated accuracy metrics. Never present them to judges as benchmark performance.
- **Grounding boxes are pixel coordinates only** — no CRS transform, no map overlay yet (Phase 10). The referring-phrase extractor (`extract_referring_phrase` in `models/grounding.py`) is rule-based and only strips a fixed set of English question scaffolding — verify it against your actual demo queries before relying on it live.
- **Rule-based routing is intentional**, not a placeholder for an LLM. The PS explicitly permits deterministic/rule-based orchestration; only replace it with a learned classifier if testing shows the rules are insufficient.
- **Do not build** (per project brief, to protect the 4-day/limited timeline): authentication, mobile apps, blockchain, AR/VR, custom satellite-data infrastructure, training a large VLM from scratch, or elaborate cloud infrastructure. Every added feature should trace back to a PS requirement, evaluation score, the adaptive-evidence story, or demo quality — if it doesn't, cut it.

---

## 9. Next actions

1. **Phase 7:** bi-temporal change detection / change-VQA specialist model — this is the other mandatory multi-image capability and currently has zero model behind it. This is the next highest-priority gap.
2. **Phase 8:** optical+SAR cross-modal fusion — required before Phase 9's cloud→SAR hero feature can do anything beyond routing.
3. **Adaptation debt:** run the VQA LoRA pipeline for real (pull the actual PS-linked BigEarthNet.txt dataset, verify its manifest matches `prepare_bigearthnet.py`'s assumptions, then run `train_lora_vqa.py`). Consider whether grounding needs its own adaptation pass too, or whether time is better spent on Phase 7/8 given only one adapted component is PS-mandatory.
4. Decide compute availability (local GPU / CPU-only / cloud) now if you haven't — Phase 5/6's models were chosen specifically because they're CPU-viable; Phase 7/8 models may need to be picked with the same constraint in mind.
