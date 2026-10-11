"""
POST /api/query

Phase 4 (P0). Satisfies PS requirements:
- "interpret the query and classify the requested task"
- "check the number, modality, format, metadata, and compatibility"
- "select one or more models or tools from a predefined registry"
- "provide an auditable execution summary containing the selected
   task, model/tool names, and key parameters"

Phase 5 added: task == vqa runs a real model (models/vqa.py).
Phase 6 added: task == grounding runs a real model (models/grounding.py) —
the second mandatory single-image task (text-guided region grounding,
chosen over captioning — see PHASE6.md §1). Phase 7 added: change_detection
and change_vqa run real handlers (models/change.py — see PHASE7.md). Phase 8
added: optical_sar runs a real handler (models/optical_sar.py — see
PHASE8.md). All six task types are now live; nothing falls through to the
generic "pending" branch anymore, though it's kept below for any future
task type.

IMPORTANT — scope boundary (unchanged since Phase 4):
Routing/validation logic never fabricates results. tool_output is only
populated once a real specialist model actually ran for that request.
Each task's model call is isolated in its own handler function below so
adding Phase 7/8 doesn't grow a single unreadable branch.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from agent.registry import TASK_REGISTRY, TaskType
from agent.router import classify_query, validate_inputs
from agent.nlp_extract import extract_location_time
from agent.followup import followup_intent, infer_target_phrase, is_followup_query
from api.upload import IMAGE_PATHS, IMAGE_REGISTRY
from evidence.confidence_map import (
    from_answer_confidence,
    from_grounding_box,
    render_change_confidence_map,
    render_optical_sar_confidence_map,
)

router = APIRouter()


class SourceRef(BaseModel):
    """Phase 13: explicit source attribution for every image that fed a
    result, separate from the free-text `note` field. Answers "which
    uploaded file(s) is this evidence based on?" in a structured way."""

    image_id: str
    filename: str
    modality: str
    role: str


def _build_sources(images: list) -> list[dict]:
    sources = []
    for m in images:
        sources.append(
            SourceRef(
                image_id=m.image_id,
                filename=m.filename,
                modality=getattr(m.modality, "value", m.modality),
                role=getattr(m.role, "value", m.role),
            ).model_dump()
        )
    return sources


class QueryRequest(BaseModel):
    query: str
    image_ids: list[str]
    # Phase 15: optional follow-up context. The frontend sends back the
    # PREVIOUS turn's query/answer text so a short referential query
    # ("Where exactly?") can lean on it — see agent/followup.py. Nothing
    # is stored server-side; this is per-request only.
    previous_query: Optional[str] = None
    previous_answer: Optional[str] = None
    # Phase 16: hops 2 ("How much area?") and 3 ("Was this visible in
    # SAR?") need more than free text — they reuse the PREVIOUS turn's
    # grounding bounding box and the image_id it was computed on. Both
    # optional; if a follow-up intent needs them and they're missing,
    # the relevant handler below returns a clear "ask a full grounding
    # question first" message rather than guessing.
    previous_box: Optional[dict] = None  # {"x_min","y_min","x_max","y_max"}
    previous_image_id: Optional[str] = None


class InputCheck(BaseModel):
    valid: bool
    issues: list[str]


class VQAOutput(BaseModel):
    answer: str
    confidence: float
    confidence_level: Literal["high", "medium", "insufficient"]
    confidence_note: str
    model_id: str
    adapted: bool
    latency_ms: int


class BoundingBoxOut(BaseModel):
    x_min: float
    y_min: float
    x_max: float
    y_max: float


class GroundingOutput(BaseModel):
    phrase: str
    found: bool
    box: Optional[BoundingBoxOut]
    score: float
    confidence_level: Literal["high", "medium", "insufficient"]
    confidence_note: str
    image_width: int
    image_height: int
    model_id: str
    adapted: bool
    latency_ms: int
    # Phase 16: which uploaded image this box was actually computed on
    # (optical or SAR — the Evidence Sufficiency Engine may have picked
    # either). Lets the frontend pass the right image_id back for
    # follow-up hop 2 (area estimate needs that image's georeferencing).
    image_id: Optional[str] = None


class ChangeBoxOut(BaseModel):
    x_min: float
    y_min: float
    x_max: float
    y_max: float


class ChangeMapOut(BaseModel):
    changed_pixel_pct: float
    change_level: Literal["high", "medium", "low"]
    change_note: str
    largest_change_box: Optional[ChangeBoxOut]
    width: int
    height: int
    method: str


class ChangeDetectionOutput(BaseModel):
    description: str
    change_map: ChangeMapOut


class ChangeVQAOutput(BaseModel):
    answer: str
    change_map: ChangeMapOut
    vqa_confidence: float
    vqa_confidence_level: Literal["high", "medium", "insufficient"]
    model_id: str
    adapted: bool


class VerificationOut(BaseModel):
    """Phase 15: second, independently-computed deterministic check —
    see evidence/verification.py's docstring for the honesty contract on
    what this is and is not (NOT a second trained model)."""

    method_a: str
    method_a_level: str
    method_b: str
    method_b_level: str
    agree: bool
    note: str


class AreaEstimateOutput(BaseModel):
    """Phase 16, follow-up hop 2 — see evidence/area.py's honesty
    contract. area_km2 is None whenever real georeferencing isn't
    available; area_note always explains why in that case."""

    target_phrase: str
    box: BoundingBoxOut
    area_km2: Optional[float]
    area_note: str
    method: str


class OpticalSarOutput(BaseModel):
    target: Literal["water", "built_up"]
    answer: str
    agreement_pct: float
    optical_only_pct: float
    sar_only_pct: float
    confidence_level: Literal["high", "medium", "low"]
    width: int
    height: int
    method: str


class QueryResponse(BaseModel):
    query: str
    task: TaskType
    task_label: str
    ps_requirement: str
    classification_reason: str
    images_considered: list[str]
    input_check: InputCheck
    selected_tool: str
    status: Literal["routed", "blocked", "insufficient_evidence", "conflicting_evidence"]
    execution_trace: list[str]
    tool_output: Optional[dict] = None
    note: str
    # Phase 13 additions:
    extracted_locations: list[str] = []
    extracted_time_expressions: list[str] = []
    nlp_extraction_note: str = ""
    sources: list[dict] = []


class _ToolRunResult:
    """Internal — carries tool_output + note + trace lines back to run_query."""

    def __init__(
        self,
        tool_output: Optional[dict],
        note: str,
        extra_trace: list[str],
        status_override: Optional[str] = None,
    ):
        self.tool_output = tool_output
        self.note = note
        self.extra_trace = extra_trace
        self.status_override = status_override


def _missing_file_result(task_label: str) -> _ToolRunResult:
    return _ToolRunResult(
        tool_output=None,
        note=(
            f"{task_label} was selected but the uploaded file could not be found on disk "
            "(likely a backend restart cleared the in-memory registry but not this "
            "reference, or the file was deleted). Re-upload the image and retry."
        ),
        extra_trace=["Tool Execution: failed — source image file missing from disk"],
    )


def _optical_sar_pair(images: list) -> tuple[Optional[Path], Optional[Path]]:
    """Phase 9: pick the single-role optical image and single-role SAR
    image out of whatever was supplied (there may be 1 or 2). Used by the
    Evidence Sufficiency Engine to decide which one to actually run the
    model on."""
    optical_meta = next((m for m in images if getattr(m.modality, "value", m.modality) == "optical"), None)
    sar_meta = next((m for m in images if getattr(m.modality, "value", m.modality) == "sar"), None)
    optical_path = IMAGE_PATHS.get(optical_meta.image_id) if optical_meta else None
    sar_path = IMAGE_PATHS.get(sar_meta.image_id) if sar_meta else None
    return optical_path, sar_path


def _optical_sar_ids(images: list) -> tuple[Optional[str], Optional[str]]:
    """Phase 16: companion to _optical_sar_pair() that returns image_ids
    instead of paths — needed so a grounding response can tell the
    frontend WHICH uploaded image its box was actually computed on
    (the Evidence Sufficiency Engine may have picked either optical or
    SAR), so follow-up hop 2 (area estimate) knows which image's
    georeferencing to use."""
    optical_meta = next((m for m in images if getattr(m.modality, "value", m.modality) == "optical"), None)
    sar_meta = next((m for m in images if getattr(m.modality, "value", m.modality) == "sar"), None)
    return (
        optical_meta.image_id if optical_meta else None,
        sar_meta.image_id if sar_meta else None,
    )


def _run_vqa(images: list, req: "QueryRequest") -> _ToolRunResult:
    from evidence.sufficiency import decide_evidence  # deferred import

    optical_path, sar_path = _optical_sar_pair(images)
    if optical_path is None and sar_path is None:
        return _missing_file_result("VQA")

    try:
        decision = decide_evidence(optical_path, sar_path)
    except ValueError:
        return _missing_file_result("VQA")

    if decision.decision == "unsupported":
        return _ToolRunResult(
            tool_output=None,
            note=(
                f"Optical evidence insufficient (cloud coverage ~{decision.cloud_pct}%) and no "
                "complementary SAR image was supplied this session. SatQuery does not guess in "
                "this case — upload a SAR image of the same area, or provide a clearer optical image."
            ),
            extra_trace=decision.trace + ["Tool Execution: skipped — insufficient evidence, no SAR fallback available"],
            status_override="insufficient_evidence",
        )

    image_path = sar_path if decision.selected_modality == "sar" else optical_path
    if image_path is None or not Path(image_path).exists():
        return _missing_file_result("VQA")

    from models.vqa import run_vqa  # deferred: heavy import only when actually needed

    trace = list(decision.trace)
    trace.append("Tool Execution: running VQA specialist model…")
    start = time.perf_counter()
    try:
        result = run_vqa(Path(image_path), req.query)
    except Exception as exc:  # noqa: BLE001 — never let a model crash take down the API
        trace.append(f"Tool Execution: failed — {exc}")
        return _ToolRunResult(
            tool_output=None,
            note=(
                "VQA model invocation raised an error. Check backend logs; the most "
                "common hackathon-week cause is missing torch/transformers/pillow "
                "dependencies — see backend/requirements.txt."
            ),
            extra_trace=trace,
        )

    latency_ms = int((time.perf_counter() - start) * 1000)
    trace.append(f"Evidence Trace: model answer generated ({latency_ms} ms)")
    trace.append(
        f"Evidence Trace: confidence {result.confidence:.2f} "
        f"({result.confidence_level.upper()}) — {result.confidence_note}"
    )
    if decision.decision == "insufficient_no_fallback":
        trace.append(
            "Evidence Trace: WARNING — this answer was generated from optical evidence already "
            "flagged insufficient, with no SAR fallback available; treat it as unreliable"
        )
    trace.append(
        "Evidence Fusion: n/a — single-model, single-image task; "
        "fusion applies once optical+SAR (Phase 8) or multi-model "
        "verification is in play"
    )
    tool_output = VQAOutput(
        answer=result.answer,
        confidence=round(result.confidence, 4),
        confidence_level=result.confidence_level,
        confidence_note=result.confidence_note,
        model_id=result.model_id,
        adapted=result.adapted,
        latency_ms=latency_ms,
    ).model_dump()
    cm = from_answer_confidence(128, 128, result.confidence, result.confidence_level)
    tool_output["confidence_map_png"] = cm.png_base64
    tool_output["confidence_map_kind"] = cm.kind
    tool_output["confidence_map_note"] = cm.note
    note = (
        "VQA answer + confidence come from a live model call (Salesforce/blip-vqa-base"
        + (" + BigEarthNet LoRA adapter" if result.adapted else ", NOT yet domain-adapted — base weights only")
        + "). Confidence is a decoding-confidence proxy, not a validated "
        "benchmark accuracy figure — do not present it as such to judges."
    )
    if decision.selected_modality == "sar":
        note = (
            f"Optical evidence was insufficient (cloud coverage ~{decision.cloud_pct}%) — this "
            "answer was generated from the SAR image instead. " + note
        )
    elif decision.decision == "insufficient_no_fallback":
        note = (
            f"Optical evidence is insufficient (cloud coverage ~{decision.cloud_pct}%) and no SAR "
            "image was available to switch to — this answer is from cloudy optical imagery and "
            "should be treated as unreliable. " + note
        )
    return _ToolRunResult(tool_output=tool_output, note=note, extra_trace=trace)


def _run_grounding(images: list, req: "QueryRequest") -> _ToolRunResult:
    from evidence.sufficiency import decide_evidence  # deferred import

    optical_path, sar_path = _optical_sar_pair(images)
    if optical_path is None and sar_path is None:
        return _missing_file_result("Grounding")

    try:
        decision = decide_evidence(optical_path, sar_path)
    except ValueError:
        return _missing_file_result("Grounding")

    if decision.decision == "unsupported":
        return _ToolRunResult(
            tool_output=None,
            note=(
                f"Optical evidence insufficient (cloud coverage ~{decision.cloud_pct}%) and no "
                "complementary SAR image was supplied this session. SatQuery does not guess in "
                "this case — upload a SAR image of the same area, or provide a clearer optical image."
            ),
            extra_trace=decision.trace + ["Tool Execution: skipped — insufficient evidence, no SAR fallback available"],
            status_override="insufficient_evidence",
        )

    image_path = sar_path if decision.selected_modality == "sar" else optical_path
    if image_path is None or not Path(image_path).exists():
        return _missing_file_result("Grounding")

    from models.grounding import extract_referring_phrase, run_grounding  # deferred import

    trace = list(decision.trace)

    # Phase 15: follow-up handling. A short referential query like "Where
    # exactly?" extracts to a near-useless literal phrase on its own — if
    # it looks like a follow-up and the frontend supplied the previous
    # turn's answer text, pull a concrete target phrase out of THAT answer
    # instead. See agent/followup.py for exactly what this is/isn't.
    effective_query = req.query
    if is_followup_query(req.query) and req.previous_answer:
        inferred = infer_target_phrase(req.previous_answer)
        if inferred:
            effective_query = f"highlight the {inferred}"
            trace.append(
                f'Follow-up Context: "{req.query}" read as a follow-up to the previous answer — '
                f'inferred target phrase "{inferred}" from it (not a memory/dialogue model — see '
                "agent/followup.py)"
            )

    phrase = extract_referring_phrase(effective_query)
    trace.append(f'Decision Trace: extracted referring phrase "{phrase}" from query')
    trace.append("Tool Execution: running grounding specialist model…")
    start = time.perf_counter()
    try:
        result = run_grounding(Path(image_path), effective_query)
    except Exception as exc:  # noqa: BLE001
        trace.append(f"Tool Execution: failed — {exc}")
        return _ToolRunResult(
            tool_output=None,
            note=(
                "Grounding model invocation raised an error. Check backend logs; the most "
                "common hackathon-week cause is missing torch/transformers/pillow "
                "dependencies — see backend/requirements.txt."
            ),
            extra_trace=trace,
        )

    latency_ms = int((time.perf_counter() - start) * 1000)
    if result.found:
        trace.append(f"Evidence Trace: region located ({latency_ms} ms)")
        trace.append(
            f"Evidence Trace: confidence {result.score:.2f} "
            f"({result.confidence_level.upper()}) — {result.confidence_note}"
        )
    else:
        trace.append(f"Evidence Trace: no matching region found ({latency_ms} ms) — {result.confidence_note}")
    if decision.decision == "insufficient_no_fallback":
        trace.append(
            "Evidence Trace: WARNING — this result was generated from optical evidence already "
            "flagged insufficient, with no SAR fallback available; treat it as unreliable"
        )
    trace.append(
        "Evidence Fusion: n/a — single-model, single-image task; "
        "fusion applies once optical+SAR (Phase 8) or multi-model verification is in play"
    )

    optical_id, sar_id = _optical_sar_ids(images)
    used_image_id = sar_id if decision.selected_modality == "sar" else optical_id

    box_out = BoundingBoxOut(**result.box.__dict__) if result.box else None
    tool_output = GroundingOutput(
        phrase=result.phrase,
        found=result.found,
        box=box_out,
        score=round(result.score, 4),
        confidence_level=result.confidence_level,
        confidence_note=result.confidence_note,
        image_width=result.image_width,
        image_height=result.image_height,
        model_id=result.model_id,
        adapted=result.adapted,
        latency_ms=latency_ms,
        image_id=used_image_id,
    ).model_dump()
    box_tuple = (result.box.x_min, result.box.y_min, result.box.x_max, result.box.y_max) if result.box else None
    cm = from_grounding_box(result.image_width, result.image_height, box_tuple, result.score)
    tool_output["confidence_map_png"] = cm.png_base64
    tool_output["confidence_map_kind"] = cm.kind
    tool_output["confidence_map_note"] = cm.note
    note = (
        f"Grounding box + score come from a live model call (google/owlvit-base-patch32"
        + (" + RS LoRA adapter" if result.adapted else ", NOT yet domain-adapted — base weights only")
        + f') on the extracted phrase "{phrase}". No map overlay yet — that lands in Phase 10; '
        "box coordinates are pixel coordinates on the original uploaded image. Score is the "
        "detector's own match confidence, not a validated grounding-accuracy benchmark figure."
    )
    if decision.selected_modality == "sar":
        note = (
            f"Optical evidence was insufficient (cloud coverage ~{decision.cloud_pct}%) — this "
            "result was generated from the SAR image instead. " + note
        )
    elif decision.decision == "insufficient_no_fallback":
        note = (
            f"Optical evidence is insufficient (cloud coverage ~{decision.cloud_pct}%) and no SAR "
            "image was available to switch to — this result is from cloudy optical imagery and "
            "should be treated as unreliable. " + note
        )
    return _ToolRunResult(tool_output=tool_output, note=note, extra_trace=trace)


def _run_area_followup(images: list, req: "QueryRequest") -> _ToolRunResult:
    """Phase 16, follow-up hop 2 ("How much area?"). Pure geometry over
    the PREVIOUS turn's grounding box — does not call any model. See
    evidence/area.py's honesty contract for exactly when a real area
    figure is/isn't produced."""
    if not req.previous_box or not req.previous_image_id:
        return _ToolRunResult(
            tool_output=None,
            note=(
                "This reads as an area follow-up (\"how much area?\"), but no previous "
                "grounding result was supplied to measure. Ask a full grounding question "
                'first (e.g. "highlight the water body"), then "how much area?".'
            ),
            extra_trace=[
                "Follow-up Context: area follow-up detected but no previous_box/"
                "previous_image_id in request — nothing to measure"
            ],
            status_override="blocked",
        )

    meta = IMAGE_REGISTRY.get(req.previous_image_id)
    if meta is None:
        return _ToolRunResult(
            tool_output=None,
            note=(
                "The image the previous grounding box was computed on is no longer known "
                "to this backend session (likely a restart). Re-run the grounding question, "
                "then ask for the area again."
            ),
            extra_trace=["Follow-up Context: previous_image_id not found in IMAGE_REGISTRY"],
            status_override="blocked",
        )

    required_keys = {"x_min", "y_min", "x_max", "y_max"}
    if not required_keys.issubset(req.previous_box.keys()):
        return _ToolRunResult(
            tool_output=None,
            note="previous_box was missing one or more of x_min/y_min/x_max/y_max — cannot compute area.",
            extra_trace=["Follow-up Context: malformed previous_box in request"],
            status_override="blocked",
        )

    from evidence.area import estimate_box_area_km2  # deferred import

    box = req.previous_box
    result = estimate_box_area_km2(
        (box["x_min"], box["y_min"], box["x_max"], box["y_max"]),
        meta.width,
        meta.height,
        meta.crs,
        meta.bounds,
    )
    phrase = infer_target_phrase(req.previous_answer or "") or "the previously grounded region"

    trace = [
        f'Follow-up Context: "{req.query}" read as an area follow-up to the previous grounding '
        f"result on image {req.previous_image_id} (agent/followup.py) — reusing its bounding box, "
        "not re-running detection",
        "Tool Execution: computing area from bounding box + source image georeferencing",
        f"Evidence Trace: {result.area_note}",
    ]

    tool_output = AreaEstimateOutput(
        target_phrase=phrase,
        box=BoundingBoxOut(**box),
        area_km2=result.area_km2,
        area_note=result.area_note,
        method=result.method,
    ).model_dump()

    note = (
        "Area estimate is a geometric calculation over the previous grounding box's pixel "
        "footprint — not a new model call, and not a segmentation of the actual object. "
        + result.area_note
    )
    return _ToolRunResult(tool_output=tool_output, note=note, extra_trace=trace)


def _run_sar_check_followup(images: list, req: "QueryRequest") -> _ToolRunResult:
    """Phase 16, follow-up hop 3 ("Was this visible in SAR?"). Re-runs
    the SAME grounding model on the SAR image supplied THIS session for
    whatever phrase the previous turn's answer implies. This is an
    independent detector call on different pixels — it does not fuse or
    jointly reason over optical+SAR the way the dedicated optical_sar
    task (Phase 8) does."""
    sar_meta = next((m for m in images if getattr(m.modality, "value", m.modality) == "sar"), None)
    if sar_meta is None:
        return _ToolRunResult(
            tool_output=None,
            note=(
                "This reads as a SAR cross-check follow-up, but no SAR image was supplied in "
                "this request — upload a SAR image of the same area (Optical/SAR slots, not "
                "Before/After) to cross-check."
            ),
            extra_trace=["Follow-up Context: SAR cross-check requested but no SAR image among image_ids"],
            status_override="blocked",
        )

    sar_path = IMAGE_PATHS.get(sar_meta.image_id)
    if sar_path is None or not Path(sar_path).exists():
        return _missing_file_result("SAR cross-check")

    phrase = infer_target_phrase(req.previous_answer or "")
    if not phrase:
        return _ToolRunResult(
            tool_output=None,
            note=(
                "Could not determine what to look for in the SAR image from the previous "
                "answer's text — ask a fuller follow-up, e.g. \"was the flooded area visible "
                'in SAR?".'
            ),
            extra_trace=[
                "Follow-up Context: SAR cross-check requested but no target phrase could be "
                "inferred from previous_answer"
            ],
            status_override="blocked",
        )

    from models.grounding import run_grounding  # deferred import

    trace = [
        f'Follow-up Context: "{req.query}" read as a SAR cross-check follow-up — re-running '
        f'grounding for "{phrase}" against the SAR image instead of the original optical image '
        "(agent/followup.py)",
        "Tool Execution: running grounding specialist model on SAR image…",
    ]
    start = time.perf_counter()
    try:
        result = run_grounding(Path(sar_path), f"highlight the {phrase}")
    except Exception as exc:  # noqa: BLE001
        trace.append(f"Tool Execution: failed — {exc}")
        return _ToolRunResult(
            tool_output=None,
            note="Grounding model invocation on the SAR image raised an error. Check backend logs.",
            extra_trace=trace,
        )

    latency_ms = int((time.perf_counter() - start) * 1000)
    if result.found:
        trace.append(
            f"Evidence Trace: region located in SAR ({latency_ms} ms), confidence "
            f"{result.score:.2f} ({result.confidence_level.upper()})"
        )
    else:
        trace.append(f"Evidence Trace: no matching region found in SAR ({latency_ms} ms) — {result.confidence_note}")
    trace.append(
        "Evidence Fusion: n/a — this is an independent detector call on the SAR image, not a "
        "joint optical+SAR fusion result; see the dedicated Optical+SAR task (Phase 8) for that."
    )

    box_out = BoundingBoxOut(**result.box.__dict__) if result.box else None
    tool_output = GroundingOutput(
        phrase=result.phrase,
        found=result.found,
        box=box_out,
        score=round(result.score, 4),
        confidence_level=result.confidence_level,
        confidence_note=result.confidence_note,
        image_width=result.image_width,
        image_height=result.image_height,
        model_id=result.model_id,
        adapted=result.adapted,
        latency_ms=latency_ms,
        image_id=sar_meta.image_id,
    ).model_dump()
    box_tuple = (result.box.x_min, result.box.y_min, result.box.x_max, result.box.y_max) if result.box else None
    cm = from_grounding_box(result.image_width, result.image_height, box_tuple, result.score)
    tool_output["confidence_map_png"] = cm.png_base64
    tool_output["confidence_map_kind"] = cm.kind
    tool_output["confidence_map_note"] = cm.note

    note = (
        (f'Also visible in SAR (score {result.score:.2f}). ' if result.found else "Not detected in SAR. ")
        + f'Same OWL-ViT grounding model (google/owlvit-base-patch32) re-run on the SAR image for "{phrase}" — '
        "an independent detector call on different pixels, not a claim that optical and SAR were "
        "jointly verified by a fusion model."
    )
    return _ToolRunResult(tool_output=tool_output, note=note, extra_trace=trace)


def _before_after_paths(images: list) -> tuple[Optional[Path], Optional[Path]]:
    """Optical-only Before/After lookup — kept for any caller that only
    cares about the optical pair."""
    before = next((m for m in images if getattr(m.role, "value", m.role) == "before"), None)
    after = next((m for m in images if getattr(m.role, "value", m.role) == "after"), None)
    before_path = IMAGE_PATHS.get(before.image_id) if before else None
    after_path = IMAGE_PATHS.get(after.image_id) if after else None
    return before_path, after_path


def _temporal_paths_by_modality(
    images: list,
) -> tuple[Optional[Path], Optional[Path], Optional[Path], Optional[Path]]:
    """Phase 12: split whatever Before/After images were supplied by
    modality, so the Evidence Sufficiency Engine can compare an optical
    pair against a SAR pair. Returns
    (optical_before, optical_after, sar_before, sar_after)."""

    def _find(role: str, modality: str) -> Optional[Path]:
        meta = next(
            (
                m
                for m in images
                if getattr(m.role, "value", m.role) == role
                and getattr(m.modality, "value", m.modality) == modality
            ),
            None,
        )
        return IMAGE_PATHS.get(meta.image_id) if meta else None

    return (
        _find("before", "optical"),
        _find("after", "optical"),
        _find("before", "sar"),
        _find("after", "sar"),
    )


def _run_change_detection(images: list, req: "QueryRequest") -> _ToolRunResult:
    from evidence.sufficiency import decide_temporal_evidence  # deferred import (Phase 11/12)

    opt_before, opt_after, sar_before, sar_after = _temporal_paths_by_modality(images)
    if opt_before is None or opt_after is None:
        return _missing_file_result("Change Detection")

    decision = decide_temporal_evidence(opt_before, opt_after, sar_before, sar_after)

    if decision.decision == "unsupported":
        return _ToolRunResult(
            tool_output=None,
            note=(
                f"Cloud coverage reaches ~{decision.worst_cloud_pct}% on at least one of the "
                "Before/After images, and there is no complete complementary SAR before/after pair "
                "to fall back on. SatQuery does not guess a change result in this case — upload a "
                "clearer pair, or a co-registered SAR Before AND After pair for the same dates."
            ),
            extra_trace=decision.trace + ["Tool Execution: skipped — insufficient temporal evidence, no SAR fallback available"],
            status_override="insufficient_evidence",
        )

    before_path, after_path = (
        (sar_before, sar_after) if decision.selected_modality == "sar" else (opt_before, opt_after)
    )
    if before_path is None or after_path is None:
        return _missing_file_result("Change Detection")

    from models.change import run_change_detection  # deferred import

    trace = list(decision.trace)
    trace.append("Tool Execution: running change-detection specialist (pixel-diff heuristic)…")
    start = time.perf_counter()
    try:
        result = run_change_detection(before_path, after_path, req.query)
    except Exception as exc:  # noqa: BLE001
        trace.append(f"Tool Execution: failed — {exc}")
        return _ToolRunResult(
            tool_output=None,
            note="Change-detection invocation raised an error. Check backend logs — see backend/requirements.txt.",
            extra_trace=trace,
        )
    latency_ms = int((time.perf_counter() - start) * 1000)
    cm = result["change_map"]
    trace.append(f"Evidence Trace: change map computed ({latency_ms} ms)")
    trace.append(f"Evidence Trace: {cm['changed_pixel_pct']}% pixels changed ({cm['change_level'].upper()})")
    if decision.decision == "insufficient_no_fallback":
        trace.append(
            "Evidence Trace: WARNING — this change result was computed from a pair with cloud "
            "coverage already flagged insufficient, with no SAR fallback available; treat it as unreliable"
        )
    trace.append("Evidence Fusion: n/a — single deterministic heuristic over the pair, no separate models to fuse")

    from evidence.verification import verify_change  # deferred import (Phase 15)

    verification = verify_change(before_path, after_path, cm["change_level"], cm["changed_pixel_pct"])
    agree_ratio = "2/2" if verification.agree else "1/2"
    trace.append(f"Verification Trace: {agree_ratio} independent methods agree — {verification.note}")

    tool_output = ChangeDetectionOutput(
        description=result["description"],
        change_map=ChangeMapOut(**cm),
    ).model_dump()
    tool_output["verification"] = VerificationOut(
        method_a=verification.method_a,
        method_a_level=verification.method_a_level,
        method_b=verification.method_b,
        method_b_level=verification.method_b_level,
        agree=verification.agree,
        note=verification.note,
    ).model_dump()
    conf_map = render_change_confidence_map(before_path, after_path)
    tool_output["confidence_map_png"] = conf_map.png_base64
    tool_output["confidence_map_kind"] = conf_map.kind
    tool_output["confidence_map_note"] = conf_map.note
    note = (
        "Change map + description come from a deterministic pixel-difference heuristic "
        "(z-score threshold on grayscale intensity), NOT a trained change-detection network. "
        "See models/change.py for why, and the change_note field for caveats on this specific result."
    )
    if decision.decision == "insufficient_sar_activated":
        note = (
            f"Optical evidence was insufficient (cloud coverage up to ~{decision.worst_cloud_pct}%) — "
            "this change result was computed from the SAR before/after pair instead. " + note
        )
    elif decision.decision == "insufficient_no_fallback":
        note = (
            f"Cloud coverage is up to ~{decision.worst_cloud_pct}% on this Before/After pair and no "
            "SAR fallback was available — treat this change result as unreliable. " + note
        )
    return _ToolRunResult(tool_output=tool_output, note=note, extra_trace=trace)


def _run_change_vqa(images: list, req: "QueryRequest") -> _ToolRunResult:
    from evidence.sufficiency import decide_temporal_evidence  # deferred import (Phase 11/12)

    opt_before, opt_after, sar_before, sar_after = _temporal_paths_by_modality(images)
    if opt_before is None or opt_after is None:
        return _missing_file_result("Change VQA")

    decision = decide_temporal_evidence(opt_before, opt_after, sar_before, sar_after)

    if decision.decision == "unsupported":
        return _ToolRunResult(
            tool_output=None,
            note=(
                f"Cloud coverage reaches ~{decision.worst_cloud_pct}% on at least one of the "
                "Before/After images, and there is no complete complementary SAR before/after pair "
                "to fall back on. SatQuery does not guess a change-VQA answer in this case — upload "
                "a clearer pair, or a co-registered SAR Before AND After pair."
            ),
            extra_trace=decision.trace + ["Tool Execution: skipped — insufficient temporal evidence, no SAR fallback available"],
            status_override="insufficient_evidence",
        )

    before_path, after_path = (
        (sar_before, sar_after) if decision.selected_modality == "sar" else (opt_before, opt_after)
    )
    if before_path is None or after_path is None:
        return _missing_file_result("Change VQA")

    from models.change import run_change_vqa  # deferred import
    from evidence.fusion import check_change_conflict  # deferred import (Phase 12)

    trace = list(decision.trace)
    trace.append("Tool Execution: running change map (heuristic) + composite-image VQA…")
    start = time.perf_counter()
    try:
        result = run_change_vqa(before_path, after_path, req.query)
    except Exception as exc:  # noqa: BLE001
        trace.append(f"Tool Execution: failed — {exc}")
        return _ToolRunResult(
            tool_output=None,
            note="Change-VQA invocation raised an error. Check backend logs — see backend/requirements.txt.",
            extra_trace=trace,
        )
    latency_ms = int((time.perf_counter() - start) * 1000)
    trace.append(f"Evidence Trace: change map computed — {result.change_map.changed_pixel_pct}% pixels changed")
    trace.append(
        f"Evidence Trace: VQA answer generated ({latency_ms} ms), confidence "
        f"{result.vqa_confidence:.2f} ({result.vqa_confidence_level.upper()})"
    )
    if decision.decision == "insufficient_no_fallback":
        trace.append(
            "Evidence Trace: WARNING — this answer was generated from a pair with cloud coverage "
            "already flagged insufficient, with no SAR fallback available; treat it as unreliable"
        )
    trace.append(
        "Evidence Fusion: composite-image VQA answer presented alongside independent pixel-diff "
        "change stats — the two are shown together, not mathematically merged into one score"
    )

    # Phase 12: CONFLICTING evidence state — cross-check the deterministic
    # change_map magnitude against the free-text VQA answer's implied
    # direction. See evidence/fusion.py for exactly what this is/isn't.
    conflict = check_change_conflict(
        result.change_map.change_level, result.change_map.changed_pixel_pct, result.answer
    )
    if conflict.conflicting:
        trace.append(f"Verification Trace: CONFLICTING — {conflict.reason}")

    from evidence.verification import verify_change  # deferred import (Phase 15)

    verification = verify_change(
        before_path, after_path, result.change_map.change_level, result.change_map.changed_pixel_pct
    )
    agree_ratio = "2/2" if verification.agree else "1/2"
    trace.append(f"Verification Trace: {agree_ratio} independent methods agree — {verification.note}")

    cm_dict = result.change_map.__dict__ | {
        "largest_change_box": result.change_map.largest_change_box.__dict__
        if result.change_map.largest_change_box else None
    }
    tool_output = ChangeVQAOutput(
        answer=result.answer,
        change_map=ChangeMapOut(**cm_dict),
        vqa_confidence=result.vqa_confidence,
        vqa_confidence_level=result.vqa_confidence_level,
        model_id=result.model_id,
        adapted=result.adapted,
    ).model_dump()
    tool_output["verification"] = VerificationOut(
        method_a=verification.method_a,
        method_a_level=verification.method_a_level,
        method_b=verification.method_b,
        method_b_level=verification.method_b_level,
        agree=verification.agree,
        note=verification.note,
    ).model_dump()
    conf_map = render_change_confidence_map(before_path, after_path)
    tool_output["confidence_map_png"] = conf_map.png_base64
    tool_output["confidence_map_kind"] = conf_map.kind
    tool_output["confidence_map_note"] = conf_map.note
    note = (
        "Answer comes from feeding a single before|after composite image into the Phase 5 "
        "BLIP-VQA model — a heuristic bridge, NOT a purpose-built change-VQA (CDVQA) model. "
        "The independent pixel-diff change_map is included so judges can sanity-check the "
        "free-text answer against a deterministic signal. Do not present either as validated "
        "CDVQA benchmark accuracy."
    )
    if decision.decision == "insufficient_sar_activated":
        note = (
            f"Optical evidence was insufficient (cloud coverage up to ~{decision.worst_cloud_pct}%) — "
            "this answer was generated from the SAR before/after pair instead. " + note
        )
    elif decision.decision == "insufficient_no_fallback":
        note = (
            f"Cloud coverage is up to ~{decision.worst_cloud_pct}% on this Before/After pair and no "
            "SAR fallback was available — treat this answer as unreliable. " + note
        )
    status_override = None
    if conflict.conflicting:
        note = f"CONFLICTING EVIDENCE: {conflict.reason} " + note
        status_override = "conflicting_evidence"
    return _ToolRunResult(tool_output=tool_output, note=note, extra_trace=trace, status_override=status_override)


def _run_optical_sar(images: list, req: "QueryRequest") -> _ToolRunResult:
    optical_img = next((m for m in images if getattr(m.modality, "value", m.modality) == "optical"), None)
    sar_img = next((m for m in images if getattr(m.modality, "value", m.modality) == "sar"), None)
    optical_path = IMAGE_PATHS.get(optical_img.image_id) if optical_img else None
    sar_path = IMAGE_PATHS.get(sar_img.image_id) if sar_img else None
    if optical_path is None or sar_path is None or not optical_path.exists() or not sar_path.exists():
        return _missing_file_result("Optical-SAR Fusion")

    from models.optical_sar import run_optical_sar  # deferred import
    from evidence.fusion import check_optical_sar_conflict  # deferred import (Phase 12)

    trace = ["Tool Execution: running optical+SAR fusion (percentile-threshold agreement)…"]
    start = time.perf_counter()
    try:
        result = run_optical_sar(optical_path, sar_path, req.query)
    except Exception as exc:  # noqa: BLE001
        trace.append(f"Tool Execution: failed — {exc}")
        return _ToolRunResult(
            tool_output=None,
            note="Optical-SAR fusion invocation raised an error. Check backend logs — see backend/requirements.txt.",
            extra_trace=trace,
        )
    latency_ms = int((time.perf_counter() - start) * 1000)
    trace.append(f"Evidence Trace: optical evidence extracted — target '{result.target}'")
    trace.append(f"Evidence Trace: SAR evidence extracted — target '{result.target}'")
    trace.append(
        f"Evidence Fusion: {result.stats.agreement_pct}% agreement between modalities "
        f"({result.stats.confidence_level.upper()} confidence) ({latency_ms} ms)"
    )

    # Phase 12: CONFLICTING evidence state — flag when optical and SAR are
    # each independently flagging substantial, largely non-overlapping
    # area for the same target. See evidence/fusion.py for the exact rule.
    conflict = check_optical_sar_conflict(
        result.stats.agreement_pct, result.stats.optical_only_pct, result.stats.sar_only_pct
    )
    if conflict.conflicting:
        trace.append(f"Verification Trace: CONFLICTING — {conflict.reason}")

    from evidence.verification import verify_optical_sar  # deferred import (Phase 15)

    verification = verify_optical_sar(
        optical_path, sar_path, result.target, result.stats.agreement_pct, result.stats.confidence_level
    )
    agree_ratio = "2/2" if verification.agree else "1/2"
    trace.append(f"Verification Trace: {agree_ratio} independent methods agree — {verification.note}")

    tool_output = OpticalSarOutput(
        target=result.target,
        answer=result.answer,
        agreement_pct=result.stats.agreement_pct,
        optical_only_pct=result.stats.optical_only_pct,
        sar_only_pct=result.stats.sar_only_pct,
        confidence_level=result.stats.confidence_level,
        width=result.width,
        height=result.height,
        method=result.method,
    ).model_dump()
    tool_output["verification"] = VerificationOut(
        method_a=verification.method_a,
        method_a_level=verification.method_a_level,
        method_b=verification.method_b,
        method_b_level=verification.method_b_level,
        agree=verification.agree,
        note=verification.note,
    ).model_dump()
    conf_map = render_optical_sar_confidence_map(optical_path, sar_path, result.target)
    tool_output["confidence_map_png"] = conf_map.png_base64
    tool_output["confidence_map_kind"] = conf_map.kind
    tool_output["confidence_map_note"] = conf_map.note
    note = (
        "Fusion result comes from a deterministic percentile-threshold agreement heuristic "
        "(bright/dark optical + high/low SAR backscatter), NOT a trained multimodal fusion "
        "network and NOT radiometrically calibrated. See models/optical_sar.py for the full "
        "disclosure. This is the backend piece the Phase 9 Evidence Sufficiency Engine will "
        "call automatically when optical evidence is judged insufficient (cloud cover, etc.)."
    )
    status_override = None
    if conflict.conflicting:
        note = f"CONFLICTING EVIDENCE: {conflict.reason} " + note
        status_override = "conflicting_evidence"
    return _ToolRunResult(tool_output=tool_output, note=note, extra_trace=trace, status_override=status_override)


# Task -> handler. Only tasks with a real specialist model integrated
# appear here; anything else falls through to the generic "pending" path.
_TASK_HANDLERS = {
    TaskType.vqa: _run_vqa,
    TaskType.grounding: _run_grounding,
    TaskType.change_detection: _run_change_detection,
    TaskType.change_vqa: _run_change_vqa,
    TaskType.optical_sar: _run_optical_sar,
    # Phase 16: follow-up chain hops 2/3 — only ever dispatched to via
    # the followup_intent() override in run_query() below, never via
    # agent/router.py's keyword classifier.
    TaskType.area_estimate: _run_area_followup,
    TaskType.sar_cross_check: _run_sar_check_followup,
}


@router.post("/query", response_model=QueryResponse)
def run_query(req: QueryRequest) -> QueryResponse:
    if not req.query.strip():
        raise HTTPException(status_code=422, detail="Query text is required.")
    if not req.image_ids:
        raise HTTPException(status_code=422, detail="At least one image_id is required.")

    images = []
    missing = []
    for iid in req.image_ids:
        meta = IMAGE_REGISTRY.get(iid)
        if meta is None:
            missing.append(iid)
        else:
            images.append(meta)
    if missing:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown image_id(s): {missing}. Upload via /api/upload first.",
        )

    # Phase 16: check follow-up intent BEFORE keyword classification.
    # This fixes a real Phase 15 bug (see agent/followup.py's module
    # docstring): classify_query()'s keyword list doesn't recognise bare
    # follow-up phrasing like "Where exactly?" as grounding intent, so
    # hop 1 would silently default to VQA and never reach the grounding
    # handler at all. Detecting intent here and setting the task
    # directly — instead of hoping keyword matching happens to agree —
    # makes all three follow-up hops reliably reachable.
    intent = followup_intent(req.query) if req.previous_answer else None
    if intent == "area":
        task, reason = TaskType.area_estimate, 'matched area follow-up pattern ("how much area?") on a previous answer'
    elif intent == "sar_check":
        task, reason = (
            TaskType.sar_cross_check,
            'matched SAR cross-check follow-up pattern ("was this visible in SAR?") on a previous answer',
        )
    elif intent == "locate":
        task, reason = (
            TaskType.grounding,
            'matched follow-up locate pattern ("where exactly?") on a previous answer — '
            "routed directly to grounding (agent/followup.py), bypassing keyword classification",
        )
    else:
        task, reason = classify_query(req.query)
    spec = TASK_REGISTRY[task]
    valid, issues = validate_inputs(task, images)
    nlp = extract_location_time(req.query)

    trace = [
        f'Query received: "{req.query}"',
        f"Decision Trace: classified as {spec['label']} ({reason})",
        f"NLP Extraction: locations={nlp.locations or '[]'} ({nlp.location_source}), "
        f"time={nlp.time_expressions or '[]'}",
        f"Input validation: {len(images)} image(s) supplied — {'OK' if valid else 'FAILED'}",
    ]

    tool_output: Optional[dict] = None
    status = "routed" if valid else "blocked"
    note = (
        f"{spec['label']} has no specialist model wired in yet — "
        "task classification + input validation + tool selection only. "
        "See README for when this task's model lands."
    )

    if not valid:
        for issue in issues:
            trace.append(f"  ✗ {issue}")
        trace.append("Tool Execution: blocked — input requirements not met")
    else:
        trace.append(f"Tool selected: {spec['tool']}")

        handler = _TASK_HANDLERS.get(task)
        if handler is not None:
            run_result = handler(images, req)
            trace.extend(run_result.extra_trace)
            tool_output = run_result.tool_output
            note = run_result.note
            if run_result.status_override:
                status = run_result.status_override
        else:
            trace.append(f"Tool Execution: pending — {spec['label']} model not yet integrated")

    return QueryResponse(
        query=req.query,
        task=task,
        task_label=spec["label"],
        ps_requirement=spec["ps_requirement"],
        classification_reason=reason,
        images_considered=[m.image_id for m in images],
        input_check=InputCheck(valid=valid, issues=issues),
        selected_tool=spec["tool"],
        status=status,
        execution_trace=trace,
        tool_output=tool_output,
        note=note,
        extracted_locations=nlp.locations,
        extracted_time_expressions=nlp.time_expressions,
        nlp_extraction_note=nlp.note,
        sources=_build_sources(images),
    )

