# SatQuery AI — Phase 11 Report

**Scope:** Generalize the Evidence Sufficiency Engine (Phase 9's hero feature) to the
bi-temporal change tasks — `change_detection` and `change_vqa` — which had zero cloud check
until now.
**Status:** ✅ Done for both tasks, within a real limitation stated up front (see §2).

---

## 1. The gap this closes

`PHASE9.md §6` named this explicitly and left it open:

> "Change detection / change-VQA / optical_sar fusion do not use this engine yet... cloud-triggered
> switching only applies to VQA and grounding right now. Extending it to the other tasks is a
> reasonable Phase 10+ addition, not done here."

Phase 10 was a UI phase and correctly didn't touch this. Through Phase 10, `change_detection` and
`change_vqa` ran on whatever Before/After pair was uploaded with **no cloud check at all** — a
heavily-obstructed pair would silently produce a change map and a confident-sounding answer with
no warning. That's the same failure mode Phase 9 fixed for VQA/grounding, just left open here.

This directly serves the master brief's own architectural principle:

> "Cloud handling is the primary demonstration of this concept, but the architecture must be
> general." (§3)

---

## 2. What was built — and its real limitation

### `backend/evidence/sufficiency.py` — new `decide_temporal_evidence()`
Runs the same `estimate_cloud_coverage()` heuristic from Phase 9 on **both** the Before and After
optical images, takes the worse of the two, and returns one of three outcomes:

| Worst cloud % | Outcome | What happens |
|---|---|---|
| < 40% | **SUFFICIENT** | Proceed with the change model as normal |
| 40–75% | **INSUFFICIENT_NO_FALLBACK** | Still runs the change model, but the trace, `note`, and result carry an explicit unreliability warning |
| ≥ 75% | **UNSUPPORTED** | Refuses to guess — `tool_output` is `null`, `status` is `"insufficient_evidence"`, same as the VQA/grounding UNSUPPORTED path |

**What this does NOT do, and why:** Phase 9's single-image engine has a fourth outcome —
`INSUFFICIENT_SAR_ACTIVATED` — because a companion SAR image can genuinely be swapped in. That
outcome does not exist here. Look at the actual upload scheme
(`frontend/src/app/page.tsx` → `SLOTS`): there is exactly one **Before** slot and one **After**
slot, and both are hard-coded `modality: "optical"`. There is no "SAR Before" / "SAR After" slot,
so when a Before/After pair is too cloudy, there is nothing in the current UI for the engine to
switch to — same honesty standard as Phase 9's own README caveat about not claiming to "fetch" a
SAR image that isn't there. This phase makes that gap **visible and honest** (a warning instead of
silence); it does not close it. Closing it for real is Phase 12 (§5).

### `backend/api/query.py` — `_run_change_detection` / `_run_change_vqa`
Both now call `decide_temporal_evidence()` before invoking `models/change.py`, exactly mirroring
the pattern `_run_vqa`/`_run_grounding` already use for `decide_evidence()`:
- Its trace lines are prepended to the execution trace.
- `unsupported` short-circuits with `tool_output: null`, `status: "insufficient_evidence"`.
- `insufficient_no_fallback` still runs the model but adds a `WARNING` trace line and prefixes
  the response `note` with the cloud-coverage caveat.

No change to `models/change.py`, `agent/registry.py`, or the frontend — this is entirely inside
the two handler functions plus the new engine function, so nothing that was already working (the
deterministic pixel-diff change map itself) was touched.

---

## 3. Example — cloudy Before/After pair, change_detection

```json
{
  "task": "change_detection",
  "status": "insufficient_evidence",
  "execution_trace": [
    "Query received: \"What changed between these two images?\"",
    "Decision Trace: classified as Bi-Temporal Change Detection",
    "Input validation: 2 image(s) supplied — OK",
    "Tool selected: change_model (pixel-diff z-score change map + rule-based description) — Phase 7",
    "Evidence Trace: BEFORE image cloud coverage estimated at 12.3% (heuristic — brightness_saturation_heuristic (deterministic, not a trained cloud model))",
    "Evidence Trace: AFTER image cloud coverage estimated at 83.7% (heuristic — brightness_saturation_heuristic (deterministic, not a trained cloud model))",
    "Decision Trace: UNSUPPORTED — cloud coverage reaches 83.7% on at least one observation, and this task has no complementary SAR before/after pair to fall back on; a reliable change result cannot be given",
    "Tool Execution: skipped — insufficient temporal evidence, no SAR fallback available"
  ],
  "tool_output": null,
  "note": "Cloud coverage reaches ~83.7% on at least one of the Before/After images, and there is no complementary SAR before/after pair to fall back on. SatQuery does not guess a change result in this case — upload a clearer pair, or (once available) a co-registered SAR before/after pair for the same dates."
}
```

---

## 4. Honesty checklist (read before demoing)

- ☐ If a change query returns `status: "insufficient_evidence"`, that's the engine correctly
  refusing — not a bug. Same rule as Phase 9's VQA/grounding UNSUPPORTED state.
- ☐ Do **not** claim change detection can "switch to SAR" the way VQA/grounding can — say
  explicitly that this task currently only detects the evidence gap, it doesn't yet have
  anywhere to fail over to.
- ☐ `insufficient_no_fallback` change results are marked unreliable in the trace and `note` —
  don't quote the change percentage or description as confident if a judge asks.

---

## 5. What's still NOT done (be honest about this in a demo)

- **No SAR-before/SAR-after upload slots.** This is the actual fix for the limitation in §2 —
  add two more upload slots (or let Before/After accept either modality and pair same-role
  optical+SAR), then extend `decide_temporal_evidence()` with a real
  `insufficient_sar_activated` branch mirroring `decide_evidence()`. Flagged as the top Phase 12
  candidate.
- **CONFLICTING state still not implemented anywhere** (named in the master brief §4B, not
  built in Phase 9 or here). Would mean cross-checking two independent signals (e.g. pixel-diff
  change map vs. change-VQA's free-text answer) and flagging disagreement rather than just
  showing both side by side as Phase 7 already does. Real value, but adds complexity for a
  state that doesn't change what's returned today — deferred, not forgotten.
- **`optical_sar` untouched** — it inherently already requires and uses both modalities, so
  there's no "insufficient → switch to X" decision to add; not in scope for this phase.
- **Remote-sensing adaptation is still unrun** — no LoRA checkpoint exists
  (`backend/models/checkpoints/vqa_lora_bigearthnet/` is empty). This remains the single
  highest-priority technical debt item per `PHASE10.md §5` and is unrelated to this phase's
  scope (evidence-engine logic, not model training). It needs a real BigEarthNet.txt manifest,
  GPU/CPU time, and someone to actually run `backend/adaptation/train_lora_vqa.py` — it cannot
  be done inside a documentation/logic phase like this one.

---

## 6. Verified this session

- `python3 -m py_compile backend/evidence/sufficiency.py backend/api/query.py` — clean, no
  syntax errors.
- Not run: the actual model paths (`models/change.py`, `evidence/cloud.py`'s `rasterio`/`numpy`
  dependency) — `torch`/`rasterio` aren't installed in this environment, same limitation
  Phase 10 hit. The new code follows the exact call pattern already proven working for
  VQA/grounding in Phases 9-10, but run it against a real cloudy Before/After pair before
  demoing to be sure.

---

## 7. Next phase

**Phase 12 candidates, ranked:**
1. SAR-before/SAR-after upload slots + real `insufficient_sar_activated` branch for change
   tasks — closes the limitation this phase deliberately left open.
2. Actually run the BigEarthNet LoRA adaptation pipeline for VQA (needs real data + compute —
   an execution task, not a code-writing one).
3. CONFLICTING evidence state.
4. Real Leaflet/OpenLayers map (click-point / draw-rectangle / draw-polygon → ask) — named in
   the master brief §4E, not yet built; Satellite View is currently an image/overlay viewer,
   not a georeferenced map.
