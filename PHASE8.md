# SatQuery AI — Phase 8 Report

**Scope:** Mandatory cross-modal pair analysis — optical + SAR complementary information extraction.
**Status:** ✅ Complete — callable end-to-end. Deterministic threshold-agreement heuristic, no trained fusion model (disclosed, not hidden — see §2). This is the specialist tool the hero feature will call, not the hero feature's decision logic itself (that's Phase 9).

---

## 1. PS requirement this phase satisfies

> "Cross-modal pair analysis: The system must extract complementary information from a co-registered optical/multispectral and SAR image pair."
> Representative query: *"Use the optical and SAR images together to identify built-up and water-covered regions."*

This is also the backend half of the master brief's **PRIMARY HERO FEATURE** (cloud obstruction → evidence insufficiency → SAR activation → multimodal evidence fusion). To be precise about scope: Phase 8 builds the fusion **analysis** so it exists to be called. The Evidence Sufficiency Engine that **decides when** to activate it automatically (cloud-coverage check, insufficiency threshold, "activating complementary sensor..." trace) is Phase 9 — not yet built. Right now, optical_sar only runs when the router classifies the query as needing both sensors explicitly (e.g. "using both sensors," "fuse optical and SAR"), same as every other task this phase.

---

## 2. Why deterministic, not a trained fusion model — read this before demoing

Same constraint as Phase 7: no pip-installable, genuinely pretrained optical+SAR fusion model meets "prefer existing pretrained models" + 4-day integration. The master brief's Day 3 guidance explicitly allows this: *"Optical features + SAR features → simple fusion/classifier → prediction."*

**What's actually running, per pixel:**
- **Optical channel:** a brightness heuristic, NOT NDWI or any validated spectral index — this pipeline doesn't guarantee a NIR band is present or which band it is, so no spectral-index claim is made. Dark pixels (bottom 25th percentile) → possible water; bright pixels (top 15th percentile) → possible built-up/bare surface.
- **SAR channel:** the standard qualitative SAR interpretation — low backscatter (bottom 25th percentile intensity) → possible water/flooded surface; high backscatter (top 15th percentile) → possible built-up/rough surface. No radiometric (sigma-naught) calibration is performed on the uploaded raster.
- **Fusion rule:** simple per-pixel agreement. Both modalities flag a pixel → high-confidence; only one → low-confidence "optical-only" or "SAR-only" signal, reported separately, not silently dropped. This directly demonstrates "extract complementary information" without a black-box model standing in for judgment.

All percentages are computed from the actual uploaded pixels at request time.

---

## 3. What was built

### `backend/models/optical_sar.py`
- `run_optical_sar(optical_path, sar_path, query)` — picks a target ("water" vs "built_up") from query keywords, computes per-modality masks, fuses by agreement, returns percentages + a plain-language answer that names the agreement and each modality's unique contribution.
- Reuses `to_uint8` from `preprocessing/raster_to_rgb.py` (same public helper Phase 7 exposed) — no duplicated stretch logic.
- Resizes mismatched pixel dimensions with the same disclosed caveat as Phase 7 (not real geometric co-registration).

### `backend/api/query.py`
- Added `_run_optical_sar` handler, registered in `_TASK_HANDLERS`.
- Resolves which uploaded image is optical vs. SAR by `modality`, not by upload order — matches how the frontend's Optical/SAR upload slots already work.

### `backend/agent/registry.py`
- `optical_sar` `tool` string updated from "not yet integrated" to the real tool name.

---

## 4. Example response (`POST /api/query`, task=optical_sar)

```json
{
  "task": "optical_sar",
  "status": "routed",
  "tool_output": {
    "target": "water",
    "answer": "Both optical and SAR evidence agree on water / flooded surface in 4.1% of the scene (HIGH confidence). Optical alone flagged an additional 1.8% (possible optical-only signal — e.g. shadow or cloud edge for water, bare soil for built-up); SAR alone flagged 2.3% (possible SAR-only signal — e.g. surface roughness unrelated to the target, or optical cloud obstruction hiding a real feature).",
    "agreement_pct": 4.1,
    "optical_only_pct": 1.8,
    "sar_only_pct": 2.3,
    "confidence_level": "high",
    "width": 1024, "height": 1024,
    "method": "percentile_threshold_agreement (deterministic, not a trained fusion model)"
  }
}
```

---

## 5. Honesty checklist (read before demoing)

- ☐ Call it "optical+SAR agreement heuristic" or "prototype fusion policy" — never "AI fusion model" unqualified.
- ☐ Don't call the optical channel "NDWI" — it's a brightness heuristic, explicitly not a spectral index.
- ☐ Don't call SAR values "backscatter in dB" — no radiometric calibration was performed; it's a percentile threshold on the stored pixel values.
- ☐ `sar_only_pct` / `optical_only_pct` are disagreement, not noise — say so if a judge asks why they're non-zero; don't imply the fusion "failed."
- ☐ This does not yet auto-activate on cloud detection — that's Phase 9. Right now it only runs when the query explicitly asks for both sensors.

---

## 6. Next phase

**Phase 9 — Evidence Sufficiency Engine.** This is where the hero demo actually comes together: cloud-coverage estimation on the optical image, an insufficiency decision, and automatic hand-off into this phase's `run_optical_sar` (or Phase 7's temporal path) when triggered — plus the Geo-Execution Trace UI copy the master brief specifies. Everything Phase 7/8 built is the tool inventory that engine will call; Phase 9 is the decision layer on top.
