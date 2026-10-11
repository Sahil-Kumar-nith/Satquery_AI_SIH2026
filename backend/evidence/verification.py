"""
Multi-Method Verification — a second, independently-computed deterministic
check that cross-validates the primary change-detection / optical-SAR
fusion result.

Phase 15. Closes a gap flagged in the Phase 14 review (master brief §13B /
uisuggestions.docx "Multi-Model Verification: 3/3 models support the
detected change"): PHASE12.md and evidence/fusion.py both explicitly
deferred this — fusion.py's CONFLICTING check only cross-checks two
numbers computed by the SAME pipeline; nothing independent existed.

HONESTY CONTRACT — read before presenting this to a judge:
- This is NOT a second trained model, and it is NOT the "3/3 models"
  multi-model consensus sketched in the master brief. There is exactly
  one image-processing pipeline behind each task (see models/change.py,
  models/optical_sar.py) — no second neural network was integrated in
  Phase 15.
- What this IS: a second classical CV heuristic that uses a genuinely
  different algorithm/statistic/granularity than the primary method, so
  the two do not share a code path and can actually disagree:
    * Change detection — primary method is a per-PIXEL z-score threshold
      (models/change.py). The independent check here is a per-BLOCK
      (patch-mean) Tukey-IQR outlier rule — different spatial scale,
      different outlier statistic.
    * Optical+SAR fusion — primary method is a fixed-percentile threshold
      per modality (models/optical_sar.py). The independent check here is
      Otsu's automatic histogram threshold (Otsu, 1979) — a different,
      data-adaptive thresholding rule.
- If a judge asks "are these two independent AI models", the honest
  answer is no — they are two different deterministic CV heuristics
  cross-checking the same uploaded pixels. Real multi-model (i.e.
  multiple independently-trained models) verification is a larger,
  separate feature and is still not built.
- Agreement/disagreement here is informational, exactly like the existing
  CONFLICTING state — it never blocks a response, only adds a trace line
  and a `verification` object to `tool_output` so the disagreement is
  visible, not hidden.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image

from preprocessing.raster_to_rgb import to_uint8

_BLOCK = 16  # patch size (pixels) for the block-level change check
_HIGH_CHANGE_PCT = 8.0
_MEDIUM_CHANGE_PCT = 2.0


@dataclass
class VerificationResult:
    method_a: str
    method_a_level: str
    method_b: str
    method_b_level: str
    agree: bool
    note: str


def _load_gray(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        band = src.read(1).astype("float32")
        if src.count >= 3:
            g = src.read(2).astype("float32")
            b = src.read(3).astype("float32")
            band = (band + g + b) / 3.0
    return to_uint8(band).astype("float32")


def _resize_to_match(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if a.shape == b.shape:
        return a, b
    h, w = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
    a2 = np.array(Image.fromarray(a.astype("uint8")).resize((w, h))).astype("float32")
    b2 = np.array(Image.fromarray(b.astype("uint8")).resize((w, h))).astype("float32")
    return a2, b2


def _level_from_pct(pct: float) -> str:
    if pct >= _HIGH_CHANGE_PCT:
        return "high"
    if pct >= _MEDIUM_CHANGE_PCT:
        return "medium"
    return "low"


def verify_change(before_path: Path, after_path: Path, primary_level: str, primary_pct: float) -> VerificationResult:
    """Independent check for change_detection / change_vqa results."""
    before = _load_gray(before_path)
    after = _load_gray(after_path)
    before, after = _resize_to_match(before, after)

    h, w = before.shape
    bh, bw = h // _BLOCK, w // _BLOCK

    if bh == 0 or bw == 0:
        # Image too small to block at this patch size — whole-image
        # fallback so this never crashes on tiny demo/benchmark images.
        pct = float(abs(after.mean() - before.mean()) > 1.0) * 100.0
    else:
        crop_h, crop_w = bh * _BLOCK, bw * _BLOCK
        b_blocks = before[:crop_h, :crop_w].reshape(bh, _BLOCK, bw, _BLOCK).mean(axis=(1, 3))
        a_blocks = after[:crop_h, :crop_w].reshape(bh, _BLOCK, bw, _BLOCK).mean(axis=(1, 3))
        diff = np.abs(a_blocks - b_blocks)
        q1, q3 = np.percentile(diff, 25), np.percentile(diff, 75)
        iqr = (q3 - q1) or 1.0
        fence = q3 + 1.5 * iqr  # Tukey's outlier fence — deliberately not a z-score
        pct = float((diff > fence).mean() * 100)

    level = _level_from_pct(pct)
    agree = level == primary_level
    note = (
        f"Primary (pixel-level z-score, models/change.py): {primary_pct:.1f}% of pixels -> "
        f"{primary_level.upper()}. Independent check (block-level Tukey-outlier, {_BLOCK}x{_BLOCK}px "
        f"patches): {pct:.1f}% of patches -> {level.upper()}. "
        + ("Both methods agree." if agree else "The two methods DISAGREE — treat this result with extra caution.")
    )
    return VerificationResult(
        method_a="pixel_diff_zscore_threshold",
        method_a_level=primary_level,
        method_b="block_mean_tukey_outlier",
        method_b_level=level,
        agree=agree,
        note=note,
    )


def _otsu_threshold(arr: np.ndarray) -> float:
    """Classic Otsu (1979) automatic bimodal threshold from the 0-255
    histogram. Deliberately independent of models/optical_sar.py's fixed
    25th/85th-percentile rule — this one adapts to each image's own
    intensity distribution instead of a hardcoded cutoff."""
    hist, _ = np.histogram(arr, bins=256, range=(0, 255))
    hist = hist.astype("float64")
    total = hist.sum()
    if total == 0:
        return 127.0
    sum_total = float(np.dot(np.arange(256), hist))
    sum_b = w_b = max_var = 0.0
    threshold = 127
    for i in range(256):
        w_b += hist[i]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += i * hist[i]
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        var_between = w_b * w_f * (m_b - m_f) ** 2
        if var_between > max_var:
            max_var = var_between
            threshold = i
    return float(threshold)


def verify_optical_sar(
    optical_path: Path, sar_path: Path, target: str, primary_agreement_pct: float, primary_level: str
) -> VerificationResult:
    """Independent check for optical_sar fusion results."""
    optical = _load_gray(optical_path)
    sar = _load_gray(sar_path)
    optical, sar = _resize_to_match(optical, sar)

    opt_t = _otsu_threshold(optical)
    sar_t = _otsu_threshold(sar)

    if target == "water":
        opt_mask = optical <= opt_t
        sar_mask = sar <= sar_t
    else:
        opt_mask = optical >= opt_t
        sar_mask = sar >= sar_t

    agreement_pct = float((opt_mask & sar_mask).mean() * 100)
    if agreement_pct >= 3.0:
        level = "high"
    elif agreement_pct >= 0.5:
        level = "medium"
    else:
        level = "low"

    agree = level == primary_level
    note = (
        f"Primary (fixed-percentile threshold, models/optical_sar.py): {primary_agreement_pct:.1f}% "
        f"agreement -> {primary_level.upper()}. Independent check (Otsu automatic threshold): "
        f"{agreement_pct:.1f}% agreement -> {level.upper()}. "
        + ("Both methods agree." if agree else "The two methods DISAGREE on confidence level — treat with extra caution.")
    )
    return VerificationResult(
        method_a="percentile_threshold_agreement",
        method_a_level=primary_level,
        method_b="otsu_automatic_threshold_agreement",
        method_b_level=level,
        agree=agree,
        note=note,
    )
