"""
Specialist tool registry.

Phase 4 (P0). Satisfies PS requirement: "select one or more models or
tools from a predefined registry."

This registry is deliberately just data — no model code lives here yet.
Each entry describes what a task NEEDS (image count / modality / role)
and WHICH tool would be invoked once Phases 5-8 wire in real specialist
models. The router (router.py) uses this to validate inputs and to
report a selected_tool name in the execution trace.

Do not add model inference code to this file — it's a lookup table.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional, TypedDict


class TaskType(str, Enum):
    vqa = "vqa"
    captioning = "captioning"
    grounding = "grounding"
    change_detection = "change_detection"
    change_vqa = "change_vqa"
    optical_sar = "optical_sar"
    # Phase 16: follow-up chain hops 2 and 3 (uisuggestions.docx "Ask
    # Follow-up" / LEFT.md §3). NOT independently PS-mandatory — these
    # only ever get selected via agent/followup.py's followup_intent(),
    # never via agent/router.py's keyword classifier, because they only
    # make sense as a continuation of a previous grounding/change turn.
    area_estimate = "area_estimate"
    sar_cross_check = "sar_cross_check"


class TaskSpec(TypedDict):
    label: str
    ps_requirement: str
    min_images: int
    max_images: int
    allowed_roles: set[str]
    allowed_modalities: set[str]
    required_modalities: Optional[set[str]]  # if set, ALL of these must be present across inputs
    tool: str


TASK_REGISTRY: dict[TaskType, TaskSpec] = {
    TaskType.vqa: {
        "label": "Visual Question Answering",
        "ps_requirement": "Single-image VQA (mandatory baseline)",
        "min_images": 1,
        # Phase 9: up to 2 allowed (optical + SAR of the same scene) so the
        # Evidence Sufficiency Engine can pick whichever is usable — the
        # model itself still only ever looks at ONE image per call.
        "max_images": 2,
        "allowed_roles": {"single"},
        "allowed_modalities": {"optical", "sar"},
        "required_modalities": None,
        "tool": "vqa_model (BLIP-VQA-base, BigEarthNet-adapted where available) — Phase 5",
    },
    TaskType.captioning: {
        "label": "Scene Captioning",
        "ps_requirement": "Additional single-image task: captioning/scene description",
        "min_images": 1,
        "max_images": 2,  # Phase 9: same optical+SAR allowance as VQA, see above
        "allowed_roles": {"single"},
        "allowed_modalities": {"optical", "sar"},
        "required_modalities": None,
        "tool": "captioning_model — not yet integrated (Phase 5)",
    },
    TaskType.grounding: {
        "label": "Text-Guided Region Grounding",
        "ps_requirement": "Additional single-image task: text-guided region grounding",
        "min_images": 1,
        "max_images": 2,  # Phase 9: same optical+SAR allowance as VQA, see above
        "allowed_roles": {"single"},
        "allowed_modalities": {"optical", "sar"},
        "required_modalities": None,
        "tool": "grounding_model (OWL-ViT open-vocabulary detector) — Phase 6",
    },
    TaskType.change_detection: {
        "label": "Bi-Temporal Change Detection",
        "ps_requirement": "Multi-image change analysis (mandatory): change description/change map",
        "min_images": 2,
        # Phase 12: up to 4 allowed — one optical Before/After pair PLUS
        # one SAR Before/After pair — so the Evidence Sufficiency Engine
        # can fail over to the SAR pair when the optical pair is cloudy,
        # the same way VQA/grounding already fail over to a single SAR
        # image. The model itself still only ever runs on ONE pair per
        # call (see evidence/sufficiency.py decide_temporal_evidence()).
        "max_images": 4,
        "allowed_roles": {"before", "after"},
        "allowed_modalities": {"optical", "sar"},
        "required_modalities": None,
        "tool": "change_model (pixel-diff z-score change map + rule-based description) — Phase 7",
    },
    TaskType.change_vqa: {
        "label": "Change-Based Visual Question Answering",
        "ps_requirement": "Multi-image change analysis (mandatory): change-based VQA",
        "min_images": 2,
        "max_images": 4,  # Phase 12: same optical+SAR before/after allowance as above
        "allowed_roles": {"before", "after"},
        "allowed_modalities": {"optical", "sar"},
        "required_modalities": None,
        "tool": "change_vqa_model (composite before|after image through BLIP-VQA + change map) — Phase 7",
    },
    TaskType.optical_sar: {
        "label": "Optical–SAR Cross-Modal Analysis",
        "ps_requirement": "Cross-modal pair analysis (mandatory): optical+SAR complementary extraction",
        "min_images": 2,
        "max_images": 2,
        "allowed_roles": {"single"},
        "allowed_modalities": {"optical", "sar"},
        "required_modalities": {"optical", "sar"},
        "tool": "optical_sar_fusion_model (percentile-threshold agreement fusion) — Phase 8",
    },
    TaskType.area_estimate: {
        "label": "Follow-up: Area Estimate",
        "ps_requirement": (
            "NOT PS-mandatory — hop 2 of the follow-up chain "
            "(uisuggestions.docx 'Ask Follow-up'), only reachable via "
            "agent/followup.py after a prior grounding turn"
        ),
        "min_images": 1,
        "max_images": 2,
        "allowed_roles": {"single"},
        "allowed_modalities": {"optical", "sar"},
        "required_modalities": None,
        "tool": "area_from_geotiff_gsd (pixel-box -> km\u00b2 via source GeoTIFF georeferencing) — Phase 16",
    },
    TaskType.sar_cross_check: {
        "label": "Follow-up: SAR Cross-Check",
        "ps_requirement": (
            "NOT PS-mandatory — hop 3 of the follow-up chain "
            "(uisuggestions.docx 'Ask Follow-up'), only reachable via "
            "agent/followup.py after a prior grounding turn"
        ),
        "min_images": 1,
        "max_images": 2,
        "allowed_roles": {"single"},
        "allowed_modalities": {"optical", "sar"},
        "required_modalities": {"sar"},
        "tool": "grounding_model (google/owlvit-base-patch32), re-run on the SAR image — Phase 16",
    },
}
