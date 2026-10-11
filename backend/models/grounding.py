"""
Text-guided region grounding specialist model.

Phase 6 (P0). Satisfies PS requirement:
- "Each solution must additionally implement either captioning/scene
   description OR text-guided region grounding." — we chose grounding
   (see README §4 rationale: stronger map/demo fit, produces the
   bounding-box evidence the PS explicitly calls for: "Highlight the
   water body referred to in the query.")

MODEL CHOICE — google/owlvit-base-patch32
- Verified supports: open-vocabulary object detection. Give it an image
  + arbitrary text phrase(s), it returns bounding boxes + a match score
  per phrase. This is exactly "text-guided region grounding" — no
  detection-head retraining needed to add new object classes.
- Modality: input = 1 RGB image + 1-N text phrases. Output = boxes
  (pixel coords) + confidence score per phrase.
- Integration complexity: LOW. Pure `transformers`
  (OwlViTProcessor/OwlViTForObjectDetection), same dependency footprint
  as Phase 5's VQA model — no new packages required.
- License: Apache 2.0 (OWL-ViT / Google). Free for hackathon/research use.
- NOT remote-sensing-native. It was trained on natural-image/web data,
  so expect it to do well on generic categories ("water", "building",
  "road", "vehicle", "vegetation") and poorly on RS-specific phrasing
  ("built-up area", "paddy field") until adapted. This model is a
  strong Phase-6 candidate for the SAME LoRA-style domain adaptation
  approach as Phase 5's VQA model — not yet built; would be a Phase 6.5
  follow-up if time allows, not required for the PS baseline.
- Confidence returned is the detector's own softmax match score for the
  best box, NOT a validated grounding-accuracy benchmark figure.
"""
from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from preprocessing.raster_to_rgb import load_as_rgb

MODEL_ID = "google/owlvit-base-patch32"

# Same confidence-bucket policy as Phase 5 (master brief §10) — kept
# consistent across specialist models so the UI doesn't need per-task
# thresholds.
_HIGH_THRESHOLD = 0.30   # OWL-ViT match scores run lower than VQA token
_MEDIUM_THRESHOLD = 0.12  # probs by nature of open-vocab detection; these
                           # are a prototype policy, not a validated cutoff.

# Rule-based phrase extraction — strips grounding-intent scaffolding off
# the user's natural-language query so OWL-ViT gets a clean noun phrase.
# Deterministic on purpose, consistent with agent/router.py's approach.
_STRIP_PREFIXES = [
    "where is the", "where is", "highlight the", "highlight", "locate the",
    "locate", "point to the", "point to", "show me the", "show me",
    "draw a box around the", "draw a box around", "mark the", "mark",
    "find the", "find",
]
_STRIP_SUFFIXES = ["referred to in the query", "in this image", "in the image"]


@dataclass
class BoundingBox:
    x_min: float
    y_min: float
    x_max: float
    y_max: float


@dataclass
class GroundingResult:
    phrase: str
    found: bool
    box: Optional[BoundingBox]
    score: float
    confidence_level: str  # "high" | "medium" | "insufficient"
    confidence_note: str
    image_width: int
    image_height: int
    model_id: str
    adapted: bool


def extract_referring_phrase(query: str) -> str:
    """Turn a grounding question into a short noun phrase for the detector.

    e.g. "Where is the water body?" -> "water body"
         "Highlight the flooded area." -> "flooded area"

    Purely rule-based (strip known prefixes/suffixes + punctuation) —
    no LLM call, consistent with the router's deterministic classification.
    If nothing matches, falls back to the raw query so grounding still
    attempts something rather than failing silently.
    """
    q = query.strip().rstrip("?.!").lower()

    for prefix in sorted(_STRIP_PREFIXES, key=len, reverse=True):
        if q.startswith(prefix):
            q = q[len(prefix):].strip()
            break

    for suffix in _STRIP_SUFFIXES:
        if q.endswith(suffix):
            q = q[: -len(suffix)].strip()

    q = re.sub(r"^(the|a|an)\s+", "", q).strip()
    return q or query.strip().rstrip("?.!")


class _GroundingModel:
    """Lazy-loaded singleton, mirrors models/vqa.py's pattern."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._processor = None
        self._model = None
        self._device = "cpu"
        self._adapted = False  # no grounding adapter exists yet — see docstring

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return

            import torch
            from transformers import OwlViTForObjectDetection, OwlViTProcessor

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._processor = OwlViTProcessor.from_pretrained(MODEL_ID)
            self._model = OwlViTForObjectDetection.from_pretrained(MODEL_ID).to(self._device).eval()

    def ground(self, image_path: Path, phrase: str) -> GroundingResult:
        self._ensure_loaded()
        import torch

        image = load_as_rgb(image_path)
        width, height = image.size

        inputs = self._processor(text=[[phrase]], images=image, return_tensors="pt").to(self._device)
        with torch.no_grad():
            outputs = self._model(**inputs)

        target_sizes = torch.tensor([[height, width]])
        results = self._processor.post_process_grounded_object_detection(
            outputs=outputs, target_sizes=target_sizes, threshold=0.0
        )[0]

        boxes, scores = results["boxes"], results["scores"]
        if len(scores) == 0:
            level, note = "insufficient", "Detector returned no candidate region at all for this phrase."
            return GroundingResult(
                phrase=phrase, found=False, box=None, score=0.0,
                confidence_level=level, confidence_note=note,
                image_width=width, image_height=height,
                model_id=self._model_id_label(), adapted=self._adapted,
            )

        best_idx = int(torch.argmax(scores))
        best_score = float(scores[best_idx])
        x_min, y_min, x_max, y_max = (float(v) for v in boxes[best_idx])
        level, note = _bucket_confidence(best_score)

        return GroundingResult(
            phrase=phrase,
            found=True,
            box=BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max),
            score=best_score,
            confidence_level=level,
            confidence_note=note,
            image_width=width,
            image_height=height,
            model_id=self._model_id_label(),
            adapted=self._adapted,
        )

    def _model_id_label(self) -> str:
        return MODEL_ID + (" + RS LoRA adapter" if self._adapted else " (base, not RS-adapted)")


def _bucket_confidence(score: float) -> tuple[str, str]:
    if score >= _HIGH_THRESHOLD:
        return "high", "Detector was strongly confident this region matches the phrase."
    if score >= _MEDIUM_THRESHOLD:
        return "medium", "Detector found a plausible region but with moderate uncertainty — verify visually."
    return "insufficient", "Best candidate region had low match confidence — treat as unreliable."


_singleton: Optional[_GroundingModel] = None
_singleton_lock = threading.Lock()


def get_grounding_model() -> _GroundingModel:
    global _singleton
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:
                _singleton = _GroundingModel()
    return _singleton


def run_grounding(image_path: Path, query: str) -> GroundingResult:
    phrase = extract_referring_phrase(query)
    return get_grounding_model().ground(image_path, phrase)
