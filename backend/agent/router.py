"""
Query/Agent router.

Phase 4 (P0). Satisfies PS requirements:
- "interpret the query and classify the requested task"
- "check the number, modality, format, metadata, and compatibility of
   the input images"

Deterministic, rule-based classification on purpose — the brief and PS
both explicitly allow this ("The initial implementation may use
deterministic rules or a lightweight classifier... Do NOT over-engineer
the agent"). Swap in a lightweight LLM/classifier later ONLY if the
rules prove insufficient in testing; don't pre-optimize this.

classify_query() and validate_inputs() are pure functions — no I/O,
no model calls — so they're trivial to unit test and to reason about
during a live demo ("why did it pick this task?").
"""
from __future__ import annotations

from agent.registry import TASK_REGISTRY, TaskType

# Ordered rule groups — first match wins. Order matters: more specific
# multi-image intents are checked before generic single-image ones so
# e.g. "what changed" doesn't fall through to plain VQA.
_OPTICAL_SAR_KEYWORDS = [
    "optical and sar", "sar and optical", "combine optical", "fuse",
    "fusion", "using both", "both sensors", "radar and optical",
    "optical, sar", "together to identify",
]

_CHANGE_KEYWORDS = [
    "chang", "increase", "decrease", "before and after", "differ",
    "compare these", "compared to", "over time",
]

_CHANGE_QUESTION_KEYWORDS = [
    "what changed", "where did", "how much", "did the", "has the",
    "was there a change",
]

_GROUNDING_KEYWORDS = [
    "where is", "highlight", "locate", "point to", "show me the",
    "draw a box", "mark the",
]

_CAPTIONING_KEYWORDS = [
    "describe", "caption", "summariz", "what does this show",
    "give a description", "overview of this image",
]


def classify_query(query: str) -> tuple[TaskType, str]:
    """Returns (task, human-readable reason). Reason is shown in the
    execution trace — it's a classification justification, not hidden
    model reasoning, so it's fine to surface."""
    q = query.lower().strip()

    if any(k in q for k in _OPTICAL_SAR_KEYWORDS):
        return TaskType.optical_sar, "matched optical+SAR fusion keywords"

    if any(k in q for k in _CHANGE_KEYWORDS):
        if any(k in q for k in _CHANGE_QUESTION_KEYWORDS):
            return TaskType.change_vqa, "matched change-question phrasing over a temporal pair"
        return TaskType.change_detection, "matched change/temporal keywords"

    if any(k in q for k in _GROUNDING_KEYWORDS):
        return TaskType.grounding, "matched region-grounding keywords"

    if any(k in q for k in _CAPTIONING_KEYWORDS):
        return TaskType.captioning, "matched scene-description keywords"

    return TaskType.vqa, "no stronger signal matched — defaulted to single-image VQA"


def validate_inputs(task: TaskType, images: list) -> tuple[bool, list[str]]:
    """images: list of UploadResponse-like objects with .modality/.role
    attributes (duck-typed to avoid a circular import on api.upload)."""
    spec = TASK_REGISTRY[task]
    issues: list[str] = []

    n = len(images)
    if n < spec["min_images"] or n > spec["max_images"]:
        if spec["min_images"] == spec["max_images"]:
            issues.append(f"{spec['label']} requires exactly {spec['min_images']} image(s); got {n}.")
        else:
            issues.append(
                f"{spec['label']} requires between {spec['min_images']} and {spec['max_images']} image(s); got {n}."
            )

    for img in images:
        modality = getattr(img, "modality", None)
        role = getattr(img, "role", None)
        modality_val = getattr(modality, "value", modality)
        role_val = getattr(role, "value", role)

        if modality_val not in spec["allowed_modalities"]:
            issues.append(f"Image {img.image_id}: modality '{modality_val}' not valid for {spec['label']}.")
        if role_val not in spec["allowed_roles"]:
            hint = (
                " Hint: re-upload it under the Optical or SAR slot instead of Before/After."
                if role_val in ("before", "after") and "single" in spec["allowed_roles"]
                else (
                    " Hint: change tasks need one image in Before and one in After — not Optical/SAR."
                    if role_val == "single" and "single" not in spec["allowed_roles"]
                    else ""
                )
            )
            issues.append(
                f"Image {img.image_id}: role '{role_val}' not valid for {spec['label']} "
                f"(expected one of {sorted(spec['allowed_roles'])}).{hint}"
            )

    if spec["required_modalities"] is not None:
        present = {getattr(getattr(i, 'modality', None), 'value', getattr(i, 'modality', None)) for i in images}
        missing = spec["required_modalities"] - present
        if missing:
            issues.append(f"{spec['label']} requires modalities {sorted(missing)} — not present in supplied images.")

    if task in (TaskType.change_detection, TaskType.change_vqa) and n == 2:
        roles_present = {getattr(getattr(i, 'role', None), 'value', getattr(i, 'role', None)) for i in images}
        if roles_present != {"before", "after"}:
            issues.append("Change tasks require exactly one 'before' image and one 'after' image.")

    return (len(issues) == 0), issues
