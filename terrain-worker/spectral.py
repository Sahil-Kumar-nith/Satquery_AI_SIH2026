"""
spectral.py — the "5-Sensor Adapter" that UploadZone.tsx has been calling
via /api/spectral this whole time, on a route and worker module that
never existed. That's why every upload failed: UploadZone defaults to
modality="auto", which unconditionally calls the detect endpoint before
processing anything — with no endpoint to answer, EVERY upload broke,
not just non-optical ones.

IMPORTANT — what this module actually is and isn't, said plainly so the
demo doesn't overclaim:
  - `detect_modality` is a lightweight HEURISTIC based on color statistics
    (saturation, channel dominance, contrast). It is NOT a trained
    classifier and will misjudge ambiguous images — it exists to save a
    click, not to be scientifically authoritative. Always let the user
    override it (UploadZone already does, via the sensor chips).
  - `adapt_image` does NOT recover real thermal/NIR signal from an RGB
    photo — that information was never captured. What it does instead is
    apply visually-motivated approximations so an image that already
    LOOKS like one of these modalities gets treated consistently:
      * thermal -> grayscale intensity remapped through an "ironbow"
        palette (the convention most thermal cameras render in)
      * nir     -> channel-boost proxy loosely mimicking a false-color
        NIR-R-G composite (real NIR needs a sensor band RGB never has)
      * pan     -> desaturated to grayscale-as-RGB (panchromatic sensors
        capture no color by definition)
      * negative -> straightforward pixel inversion
  - Be upfront about this distinction in any demo or write-up: this is a
    preprocessing/visualization convenience layer, not a spectral
    reconstruction algorithm.
"""

import base64
import io
import logging
from enum import Enum

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from PIL import Image

logger = logging.getLogger("depthwizard.spectral")
router = APIRouter()


class Modality(str, Enum):
    OPTICAL = "optical"
    THERMAL = "thermal"
    NIR = "nir"
    PAN = "pan"
    NEGATIVE = "negative"


VALID_MODALITIES = {m.value for m in Modality}

# Approximate control points for the "ironbow" palette used by most
# consumer thermal cameras (black -> purple -> red -> orange -> white).
_IRONBOW_STOPS = [
    (0.00, (0, 0, 0)),
    (0.20, (30, 0, 100)),
    (0.40, (130, 0, 120)),
    (0.60, (220, 60, 0)),
    (0.80, (255, 170, 0)),
    (1.00, (255, 255, 200)),
]


def _build_ironbow_lut() -> np.ndarray:
    """256-entry RGB lookup table, linearly interpolated between the stops above."""
    lut = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        t = i / 255
        for j in range(len(_IRONBOW_STOPS) - 1):
            t0, c0 = _IRONBOW_STOPS[j]
            t1, c1 = _IRONBOW_STOPS[j + 1]
            if t0 <= t <= t1:
                f = (t - t0) / max(t1 - t0, 1e-6)
                lut[i] = [int(c0[k] + f * (c1[k] - c0[k])) for k in range(3)]
                break
        else:
            lut[i] = _IRONBOW_STOPS[-1][1]
    return lut


_IRONBOW_LUT = _build_ironbow_lut()


def detect_modality(img: Image.Image) -> tuple[Modality, float]:
    """Heuristic classifier. Returns (best guess, confidence 0-1)."""
    rgb = np.array(img.convert("RGB"), dtype=np.float32)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)
    saturation = float(np.mean((max_c - min_c) / (max_c + 1e-6)))
    std_val = float(np.std(rgb))

    if saturation < 0.05:
        # Effectively grayscale — can't reliably tell thermal-exported-
        # as-grayscale from true panchromatic without more context, so
        # default to the more literal reading.
        return (Modality.PAN, 0.55)

    if np.mean(r) > np.mean(g) * 1.3 and np.mean(r) > np.mean(b) * 1.3:
        # Red-channel dominance is a weak proxy for false-color NIR
        # composites (vegetation reflects NIR strongly). Deliberately
        # low confidence — this is the shakiest of the heuristics.
        return (Modality.NIR, 0.40)

    if saturation > 0.35 and std_val > 60:
        # Thermal palettes (ironbow/rainbow) are highly saturated with
        # strong warm-cool contrast.
        return (Modality.THERMAL, 0.45)

    return (Modality.OPTICAL, 0.70)


def adapt_image(img: Image.Image, modality: Modality) -> Image.Image:
    """Applies the modality-specific transform described in the module docstring."""
    img = img.convert("RGB")

    if modality == Modality.OPTICAL:
        return img

    if modality == Modality.PAN:
        gray = np.array(img.convert("L"), dtype=np.uint8)
        return Image.fromarray(np.stack([gray] * 3, axis=-1))

    if modality == Modality.NEGATIVE:
        inverted = 255 - np.array(img, dtype=np.uint8)
        return Image.fromarray(inverted)

    if modality == Modality.THERMAL:
        gray = np.array(img.convert("L"), dtype=np.uint8)
        colored = _IRONBOW_LUT[gray]
        return Image.fromarray(colored.astype(np.uint8))

    if modality == Modality.NIR:
        arr = np.array(img, dtype=np.float32)
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        boosted = np.stack(
            [
                np.clip(g * 1.2, 0, 255),  # NIR channel proxy <- boosted green
                np.clip(r * 1.1, 0, 255),  # red channel <- original red
                np.clip(b * 0.6, 0, 255),  # green channel <- suppressed blue
            ],
            axis=-1,
        )
        return Image.fromarray(boosted.astype(np.uint8))

    raise ValueError(f"Unhandled modality: {modality}")


def _image_to_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


@router.post("/spectral")
async def spectral_endpoint(
    action: str = Form(...),
    file: UploadFile = File(...),
    modality: str | None = Form(default=None),
):
    """
    action="detect" -> {"modality": "...", "confidence": 0.0-1.0}
    action="adapt"  -> {"modality_used": "...", "image_base64": "data:image/png;base64,..."}
    """
    raw = await file.read()
    try:
        img = Image.open(io.BytesIO(raw))
    except Exception as e:
        raise HTTPException(400, f"Could not read image: {e}")

    if action == "detect":
        best, confidence = detect_modality(img)
        logger.info(f"Detected modality={best.value} confidence={confidence}")
        return {"modality": best.value, "confidence": round(confidence, 2)}

    if action == "adapt":
        if not modality:
            raise HTTPException(400, "'modality' is required for action=adapt")
        if modality not in VALID_MODALITIES:
            raise HTTPException(400, f"Unknown modality '{modality}', expected one of {sorted(VALID_MODALITIES)}")

        adapted = adapt_image(img, Modality(modality))
        logger.info(f"Adapted image for modality={modality}")
        return {"modality_used": modality, "image_base64": _image_to_data_url(adapted)}

    raise HTTPException(400, f"Unknown action '{action}', expected 'detect' or 'adapt'")