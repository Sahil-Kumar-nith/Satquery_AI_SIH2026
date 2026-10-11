"""
Single-image VQA specialist model.

Phase 5 (P0). Satisfies PS requirement:
- "Single-image visual question answering shall be mandatory."
- "At least one visual or vision-language component must be fine-tuned
   or otherwise adapted using BigEarthNet.txt or any open source
   training data." (this is that component)

MODEL CHOICE — Salesforce/blip-vqa-base
- Verified supports: general open-domain VQA (image + free-text
  question -> free-text answer). It is NOT remote-sensing-native out of
  the box — that's why the PS-mandated adaptation step in
  backend/adaptation/ exists (see adaptation/README.md).
- Modality: input = 1 RGB image + 1 text question. Output = short text
  answer (BLIP-VQA is trained as generative, not classification-head).
- Integration complexity: LOW. Pure `transformers`, no custom CUDA ops,
  ~385M params — runs on CPU (slow, seconds/query) or any small GPU.
  This is why it was chosen over a 7B-parameter RS-VLM (e.g. GeoChat):
  those need real GPU infra we cannot assume the team has in 4 days.
- License: BSD-3-Clause (Salesforce/BLIP). Free for hackathon/research
  use. Verify before any commercial deployment.
- We are NOT claiming this is a validated remote-sensing benchmark
  model. Confidence returned below is a decoding-confidence proxy from
  the model's own generation, not a calibrated accuracy figure — see
  VQAResult.confidence_note.

ADAPTATION
If a LoRA adapter trained by backend/adaptation/train_lora_vqa.py exists
at VQA_LORA_ADAPTER_PATH, it is loaded on top of the base model. If not,
the base (non-adapted) model is used and `adapted=False` is reported
honestly in every response — never silently claim adaptation happened.
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from preprocessing.raster_to_rgb import load_as_rgb

BASE_MODEL_ID = "Salesforce/blip-vqa-base"
DEFAULT_ADAPTER_DIR = Path(__file__).resolve().parent / "checkpoints" / "vqa_lora_bigearthnet"
ADAPTER_PATH = Path(os.environ.get("VQA_LORA_ADAPTER_PATH", DEFAULT_ADAPTER_DIR))

# Buckets mirror master brief §10: HIGH / MEDIUM / INSUFFICIENT confidence.
# Thresholds are a prototype policy, not a scientifically validated cutoff
# — state that explicitly wherever this is surfaced.
_HIGH_THRESHOLD = 0.75
_MEDIUM_THRESHOLD = 0.40


@dataclass
class VQAResult:
    answer: str
    confidence: float
    confidence_level: str  # "high" | "medium" | "insufficient"
    confidence_note: str
    model_id: str
    adapted: bool


def _load_adapter_info(adapter_path: Path) -> Optional[dict]:
    """Phase 14: read the honest per-epoch/best-checkpoint metrics that
    train_lora_vqa.py wrote alongside the adapter (adapter_info.json).
    Missing/unreadable file is not an error — older adapters trained
    before this existed simply won't have one."""
    info_path = adapter_path / "adapter_info.json"
    if not info_path.exists():
        return None
    try:
        import json
        return json.loads(info_path.read_text())
    except Exception:  # noqa: BLE001
        return None


class _VQAModel:
    """Lazy-loaded singleton so the model loads once per process, not per request."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._processor = None
        self._model = None
        self._adapted = False
        self._adapter_info: Optional[dict] = None
        self._device = "cpu"

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:  # re-check inside the lock
                return

            import torch
            from transformers import BlipForQuestionAnswering, BlipProcessor

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._processor = BlipProcessor.from_pretrained(BASE_MODEL_ID)
            model = BlipForQuestionAnswering.from_pretrained(BASE_MODEL_ID)

            if ADAPTER_PATH.exists():
                try:
                    from peft import PeftModel

                    model = PeftModel.from_pretrained(model, str(ADAPTER_PATH))
                    self._adapted = True
                    self._adapter_info = _load_adapter_info(ADAPTER_PATH)
                except Exception as exc:  # noqa: BLE001
                    # Fail open to the base model — a broken adapter should
                    # never take the whole VQA endpoint down mid-hackathon.
                    print(f"[vqa] Failed to load LoRA adapter at {ADAPTER_PATH}: {exc}. "
                          f"Falling back to base {BASE_MODEL_ID}.")
                    self._adapted = False

            self._model = model.to(self._device).eval()

    def answer(self, image_path: Path, question: str) -> VQAResult:
        self._ensure_loaded()
        import torch

        image = load_as_rgb(image_path)
        inputs = self._processor(image, question, return_tensors="pt").to(self._device)

        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=30,
                output_scores=True,
                return_dict_in_generate=True,
            )

        answer_text = self._processor.decode(out.sequences[0], skip_special_tokens=True).strip()
        confidence = _decode_confidence(out)
        level, note = _bucket_confidence(confidence)

        return VQAResult(
            answer=answer_text or "(model returned an empty answer)",
            confidence=confidence,
            confidence_level=level,
            confidence_note=note,
            model_id=f"{BASE_MODEL_ID}" + (
                f" + BigEarthNet LoRA adapter (best val eval_loss={self._adapter_info['best_eval_loss']:.4f}, "
                f"epoch {self._adapter_info.get('best_epoch')})"
                if self._adapted and self._adapter_info and self._adapter_info.get("best_eval_loss") is not None
                else " + BigEarthNet LoRA adapter" if self._adapted else " (base, not yet adapted)"
            ),
            adapted=self._adapted,
        )


def _decode_confidence(generate_output) -> float:
    """Mean per-token generation probability, normalized to [0,1].

    This is a decoding-confidence proxy (how sure the model was about the
    tokens it emitted), NOT a calibrated task-accuracy estimate. Do not
    present it as validated benchmark performance.
    """
    import torch

    scores = getattr(generate_output, "scores", None)
    if not scores:
        return 0.5  # no score info available — neutral default, not a real signal

    token_probs = []
    for step_logits in scores:
        probs = torch.softmax(step_logits[0], dim=-1)
        token_probs.append(probs.max().item())

    if not token_probs:
        return 0.5
    return float(sum(token_probs) / len(token_probs))


def _bucket_confidence(confidence: float) -> tuple[str, str]:
    if confidence >= _HIGH_THRESHOLD:
        return "high", "Model was consistently confident across generated tokens."
    if confidence >= _MEDIUM_THRESHOLD:
        return "medium", "Model showed moderate token-level uncertainty — treat as indicative, not definitive."
    return "insufficient", "Model showed high token-level uncertainty — reliable analysis cannot be assumed from this alone."


_singleton: Optional[_VQAModel] = None
_singleton_lock = threading.Lock()


def get_vqa_model() -> _VQAModel:
    global _singleton
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:
                _singleton = _VQAModel()
    return _singleton


def run_vqa(image_path: Path, question: str) -> VQAResult:
    return get_vqa_model().answer(image_path, question)
