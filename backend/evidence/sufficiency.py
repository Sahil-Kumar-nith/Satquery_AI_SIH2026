"""
Evidence Sufficiency Engine — the actual "Adaptive Evidence Acquisition"
hero feature from the master brief.

Phase 9 (P0). Decides whether the optical image gives enough evidence to
answer a single-image query (VQA / grounding), and if not, switches to
the SAR image if one was already uploaded in this session.

IMPORTANT — what this does NOT do: it does not fetch, search, or acquire
a new SAR observation from any external archive/API. There is no real
satellite-data feed wired into this hackathon prototype. "SAR activation"
here means: a SAR image the user already uploaded to this session is
picked up and used instead of the cloudy optical one. If no SAR image
was uploaded, the engine cannot conjure one — it says so honestly
(INSUFFICIENT_NO_FALLBACK / UNSUPPORTED) rather than pretending.

Implements the four outcomes named in the master brief §4B:
  SUFFICIENT              -> optical evidence is fine, use it as-is
  INSUFFICIENT (+SAR)     -> optical insufficient, SAR available -> use SAR
  INSUFFICIENT (no SAR)   -> optical insufficient, no SAR -> use optical
                              anyway but flag the answer as unreliable
  UNSUPPORTED             -> cloud cover extreme AND no SAR -> refuse to
                              guess, tell the user what to upload instead
CONFLICTING (cross-checking two models against each other) is out of
scope for this phase — noted as a Phase-later item, not silently dropped.

Thresholds below are prototype policy, not scientifically validated
cutoffs — same disclosure standard as every other confidence bucket in
this codebase.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

from evidence.cloud import estimate_cloud_coverage

_INSUFFICIENT_CLOUD_PCT = 40.0  # optical judged insufficient at/above this
_UNSUPPORTED_CLOUD_PCT = 75.0   # + no SAR fallback -> refuse to guess


@dataclass
class SufficiencyDecision:
    decision: Literal[
        "sufficient", "insufficient_sar_activated", "insufficient_no_fallback", "unsupported"
    ]
    selected_modality: Literal["optical", "sar"]
    cloud_pct: Optional[float]
    cloud_note: Optional[str]
    trace: list[str]


def decide_evidence(
    optical_path: Optional[Path],
    sar_path: Optional[Path],
) -> SufficiencyDecision:
    if optical_path is None or not Path(optical_path).exists():
        # Nothing to judge — no optical image at all. If SAR exists, just
        # use it directly; no cloud check is meaningful without an
        # optical image to check.
        if sar_path is not None and Path(sar_path).exists():
            return SufficiencyDecision(
                decision="sufficient",
                selected_modality="sar",
                cloud_pct=None,
                cloud_note=None,
                trace=["Evidence Trace: no optical image supplied — using SAR directly, no cloud check needed"],
            )
        raise ValueError("decide_evidence() called with no usable optical or SAR image.")

    est = estimate_cloud_coverage(Path(optical_path))
    trace = [f"Evidence Trace: cloud coverage estimated at {est.cloud_pct}% (heuristic — {est.method})"]

    if est.cloud_pct < _INSUFFICIENT_CLOUD_PCT:
        trace.append("Decision Trace: optical evidence SUFFICIENT — proceeding with optical image")
        return SufficiencyDecision("sufficient", "optical", est.cloud_pct, est.note, trace)

    trace.append(
        f"Decision Trace: optical evidence INSUFFICIENT "
        f"(cloud coverage {est.cloud_pct}% ≥ {_INSUFFICIENT_CLOUD_PCT}% threshold)"
    )

    if sar_path is not None and Path(sar_path).exists():
        trace.append("Evidence Trace: complementary SAR observation available in this session")
        trace.append("Tool Execution: SAR activation — switching analysis to the uploaded SAR image")
        return SufficiencyDecision("insufficient_sar_activated", "sar", est.cloud_pct, est.note, trace)

    if est.cloud_pct >= _UNSUPPORTED_CLOUD_PCT:
        trace.append(
            f"Decision Trace: UNSUPPORTED — cloud coverage {est.cloud_pct}% is extreme and no "
            "complementary SAR image was supplied this session; a reliable answer cannot be given "
            "from optical alone"
        )
        return SufficiencyDecision("unsupported", "optical", est.cloud_pct, est.note, trace)

    trace.append(
        "Decision Trace: no SAR image supplied this session — proceeding with optical anyway, "
        "but this answer should be treated as unreliable given the evidence gap"
    )
    return SufficiencyDecision("insufficient_no_fallback", "optical", est.cloud_pct, est.note, trace)


# --------------------------------------------------------------------------
# Phase 11 — generalizing the engine to bi-temporal change tasks.
#
# Phases 1-10 ran change_detection/change_vqa on whatever Before/After pair
# was uploaded with NO cloud check at all — silently trusting cloudy
# imagery, unlike VQA/grounding which have had this check since Phase 9.
# PHASE9.md §6 named this gap explicitly: "Change detection / change-VQA /
# optical_sar fusion do not use this engine yet."
#
# Phase 12 — SAR-before/SAR-after fallback.
#
# Phase 11 explicitly left this open because the upload scheme only had
# one Before slot and one After slot, both hard-coded to optical, so there
# was nothing to fail over to. That upload-scheme gap is now closed on the
# frontend (frontend/src/app/page.tsx SLOTS gained sar_before/sar_after)
# and on the backend registry (agent/registry.py's change_detection /
# change_vqa max_images raised from 2 to 4 so an optical pair AND a SAR
# pair can be supplied in the same request). This function now mirrors
# decide_evidence()'s four-outcome shape for the temporal case:
#   SUFFICIENT                -> optical pair is fine, use it
#   INSUFFICIENT_SAR_ACTIVATED -> optical pair insufficient, SAR before+
#                                 after pair available -> switch to SAR
#   INSUFFICIENT_NO_FALLBACK   -> optical pair insufficient, no full SAR
#                                 pair available -> use optical anyway,
#                                 flagged unreliable
#   UNSUPPORTED                -> cloud cover extreme AND no SAR pair ->
#                                 refuse to guess
#
# IMPORTANT — SAR activation here requires BOTH a SAR "before" and a SAR
# "after" image (same honesty standard as everywhere else in this
# codebase): a single SAR image cannot support a bi-temporal comparison.
# If only one of the two SAR slots was filled, that is treated the same
# as "no SAR fallback available" — the trace says so explicitly rather
# than silently ignoring the half-supplied pair.
# --------------------------------------------------------------------------


@dataclass
class TemporalSufficiencyDecision:
    decision: Literal[
        "sufficient", "insufficient_sar_activated", "insufficient_no_fallback", "unsupported"
    ]
    selected_modality: Literal["optical", "sar"]
    before_cloud_pct: Optional[float]
    after_cloud_pct: Optional[float]
    worst_cloud_pct: Optional[float]
    trace: list[str]


def decide_temporal_evidence(
    before_path: Optional[Path],
    after_path: Optional[Path],
    sar_before_path: Optional[Path] = None,
    sar_after_path: Optional[Path] = None,
) -> TemporalSufficiencyDecision:
    """Cloud-coverage check across a Before/After optical pair, with an
    optional SAR before/after pair to fail over to (Phase 12). Backward
    compatible: if no SAR paths are supplied, behaves exactly like the
    Phase 11 version (SUFFICIENT / INSUFFICIENT_NO_FALLBACK / UNSUPPORTED
    only)."""
    before_est = estimate_cloud_coverage(Path(before_path)) if before_path and Path(before_path).exists() else None
    after_est = estimate_cloud_coverage(Path(after_path)) if after_path and Path(after_path).exists() else None

    trace: list[str] = []
    if before_est:
        trace.append(
            f"Evidence Trace: BEFORE image cloud coverage estimated at {before_est.cloud_pct}% "
            f"(heuristic — {before_est.method})"
        )
    if after_est:
        trace.append(
            f"Evidence Trace: AFTER image cloud coverage estimated at {after_est.cloud_pct}% "
            f"(heuristic — {after_est.method})"
        )

    candidates = [e.cloud_pct for e in (before_est, after_est) if e is not None]
    worst = max(candidates) if candidates else None
    before_pct = before_est.cloud_pct if before_est else None
    after_pct = after_est.cloud_pct if after_est else None

    sar_pair_available = (
        sar_before_path is not None and Path(sar_before_path).exists()
        and sar_after_path is not None and Path(sar_after_path).exists()
    )

    if worst is None:
        trace.append("Decision Trace: no optical cloud estimate available — proceeding without an evidence check")
        return TemporalSufficiencyDecision("sufficient", "optical", before_pct, after_pct, None, trace)

    if worst < _INSUFFICIENT_CLOUD_PCT:
        trace.append(
            f"Decision Trace: temporal evidence SUFFICIENT — highest cloud coverage {worst}% "
            f"is below the {_INSUFFICIENT_CLOUD_PCT}% threshold"
        )
        return TemporalSufficiencyDecision("sufficient", "optical", before_pct, after_pct, worst, trace)

    trace.append(
        f"Decision Trace: optical temporal evidence INSUFFICIENT "
        f"(cloud coverage up to {worst}% ≥ {_INSUFFICIENT_CLOUD_PCT}% threshold)"
    )

    if sar_pair_available:
        trace.append("Evidence Trace: complementary SAR before/after pair available in this session")
        trace.append("Tool Execution: SAR activation — switching change analysis to the uploaded SAR before/after pair")
        return TemporalSufficiencyDecision(
            "insufficient_sar_activated", "sar", before_pct, after_pct, worst, trace
        )

    if sar_before_path is not None or sar_after_path is not None:
        trace.append(
            "Evidence Trace: only one of the two SAR slots (Before/After) was supplied — "
            "a bi-temporal SAR fallback needs both, so this cannot be used"
        )

    if worst >= _UNSUPPORTED_CLOUD_PCT:
        trace.append(
            f"Decision Trace: UNSUPPORTED — cloud coverage reaches {worst}% on at least one "
            "observation, and no complete SAR before/after pair was supplied this session to fall "
            "back on; a reliable change result cannot be given"
        )
        return TemporalSufficiencyDecision("unsupported", "optical", before_pct, after_pct, worst, trace)

    trace.append(
        f"Decision Trace: INSUFFICIENT (cloud coverage up to {worst}%) — no complete SAR "
        "before/after pair supplied this session to switch to; proceeding with the optical pair "
        "anyway, but this change result should be treated as unreliable"
    )
    return TemporalSufficiencyDecision(
        "insufficient_no_fallback", "optical", before_pct, after_pct, worst, trace
    )
