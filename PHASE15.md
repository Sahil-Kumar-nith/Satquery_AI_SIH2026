# PHASE 15 — Multi-method verification, follow-up queries, adaptation tooling

Scope for this phase: close three of the gaps flagged in the Phase 14
review (`LEFT.md` has the full list) that were **actually buildable
without external network access** — this sandbox's egress is restricted
to package registries only (pypi/npm/github release assets); it cannot
reach `huggingface.co`, `arxiv.org`, or any dataset host. That constraint
ruled out the two highest-priority open items (running real RS adaptation
training, running a real public-benchmark evaluation) — see `LEFT.md` for
exactly why and what to do about it once you have real network/dataset
access.

## 1. Multi-Method Verification (`backend/evidence/verification.py`)

Closes: "Multi-model verification / consensus checking — not built"
(flagged against master brief §13B / uisuggestions.docx's "3/3 models
support the detected change").

**What it actually is:** a second, independently-computed deterministic
check for `change_detection`/`change_vqa` and `optical_sar` results —
not a second trained model.

- Change tasks: primary method (`models/change.py`) is a per-pixel
  z-score threshold. The new independent check is a per-block
  (16×16px patch-mean) Tukey-IQR outlier rule — different spatial scale,
  different outlier statistic, same uploaded pixels.
- Optical+SAR: primary method (`models/optical_sar.py`) is a fixed
  25th/85th-percentile threshold per modality. The new independent check
  is Otsu's automatic histogram threshold (Otsu, 1979) — adapts to each
  image's own intensity distribution instead of a hardcoded cutoff.

Wired into `POST /api/query`: every `change_detection`, `change_vqa`, and
`optical_sar` `tool_output` now carries a `verification` object
(`method_a`, `method_a_level`, `method_b`, `method_b_level`, `agree`,
`note`), and the execution trace gets a `Verification Trace: 2/2 (or 1/2)
independent methods agree — ...` line. Disagreement is informational
only — it is surfaced, never blocks the response (same philosophy as the
existing Phase 12 CONFLICTING state).

Frontend: `ChatPanel.tsx` renders a `VerificationBadge` under the
change/optical-SAR result cards — labeled "independent methods", never
"models" or "AI verified".

**Read `verification.py`'s docstring before demoing this** — if a judge
asks "are these two independent AI models", the honest answer is no.

## 2. Follow-up query context (`backend/agent/followup.py`)

Closes: "Ask follow-up multi-turn agentic reasoning — not built"
(uisuggestions.docx: "demonstrates agentic multi-step reasoning much
better than a collection of independent buttons").

**What it actually is:** two rule-based helpers, not conversational
memory and not an LLM call.

- `is_followup_query()` — pattern/length heuristic: does a query like
  "Where exactly?" or "How much area?" look like it's leaning on the
  previous turn rather than standing alone?
- `infer_target_phrase()` — pulls a short noun phrase out of the
  **previous turn's answer text** (checking the PS's own representative
  vocabulary — "built-up area", "water body", "flooded area", etc. —
  before a generic fallback).

`QueryRequest` gained two optional fields: `previous_query`,
`previous_answer`. The frontend (`ChatPanel.tsx`) now tracks the last
completed turn and sends its query + answer text back on every
subsequent request (`lib/api.ts`'s `runQuery` gained a `followup`
parameter). Nothing is stored server-side beyond the current request —
this is intentionally NOT a session store.

Wired into `_run_grounding` in `api/query.py`: if the current query looks
like a follow-up and a previous answer was supplied, the phrase fed to
the grounding model is inferred from that previous answer instead of the
(near-useless) literal text of "Where exactly?". A
`Follow-up Context: ...` execution-trace line documents when this fires,
so it's never a silent override.

**Example this enables:** ask *"Has the built-up area increased?"* (→
change_vqa), then just *"Where exactly?"* (→ grounding, auto-targeting
"built-up area" without retyping it) — the closest thing to the
brainstormed "Ask Follow-up" chain (change → grounding → ...) that's
honestly buildable without a real multi-turn planner.

**What this does NOT do:** it does not chain area calculation or SAR
cross-checking automatically (the "How much? → Was this visible in SAR?"
full chain from `uisuggestions.docx`) — only the change/VQA → grounding
handoff is wired. Extending the pattern to the other two hops is
mechanical if this proves useful in testing.

## 3. Adaptation manifest validator (`backend/adaptation/verify_manifest.py`)

Closes part of the "adaptation debt" documented since Phase 5/12 —
`adaptation/README.md` has said since Phase 5 that
`prepare_bigearthnet.py`'s column assumptions (`caption` /
`labels`/`label_1..n`) are "a reasonable guess ... not a verified
schema". This script is the five-second check that should run BEFORE
burning a training run finding that out.

`verify_manifest.py --manifest <path>` parses a local CSV/JSONL manifest
(no network, no model, no training) and reports how many rows
`prepare_bigearthnet.py` would actually find usable, listing the columns
it saw so a schema mismatch is obvious immediately. Exit code 0/1 so it's
CI/shell-script-safe ahead of the real run.

**This was smoke-tested against synthetic manifests in this sandbox**
(one matching the expected schema, one deliberately not) — both produced
the correct PASS/FAIL verdict. **It was NOT tested against the real
BigEarthNet.txt manifest**, because that dataset was not reachable from
this sandbox (see `LEFT.md`).

## What this phase deliberately did NOT attempt

- **Did not run the LoRA adaptation training pipeline for real.**
  `huggingface.co` (where `Salesforce/blip-vqa-base`'s weights and
  tokenizer live, and where the PS-linked dataset page is hosted via
  `arxiv.org`) is not reachable from this sandbox's network egress list.
  `adapted: false` remains the honest, correct state.
- **Did not run any public-benchmark evaluation** (RSVQA / VRSBench /
  CDVQA) — same network constraint; those datasets are not reachable
  here either. `BenchmarksView` still shows no fabricated numbers.
- **Did not fake or backfill either of the above with synthetic data
  labeled as real.** That would contradict every honesty disclosure
  already in this codebase (see `models/vqa.py`, `adaptation/README.md`).

See `LEFT.md` for the full, plain-language rundown of what's left and
why, written for a non-technical reader.
