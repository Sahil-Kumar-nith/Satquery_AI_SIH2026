"""
Pull a SMALL subset of GFM-Bench/BigEarthNet via Hugging Face streaming
(no 59GB/119GB bulk download) and write it out as:

    data/bigearthnet_subset/images/patch_00000.png ...
    data/bigearthnet_subset/manifest.csv   (image_path,labels)

...ready to feed straight into prepare_bigearthnet.py.

Run this from backend/adaptation/ (see chat — paths assume that).

Confirmed schema (from the HF dataset card):
    optical -> Sentinel-2 image, 12 bands, 120x120
    radar   -> Sentinel-1 image (not used here — VQA path is optical-only for now)
    label   -> multi-label classification target over the 19 CLASS_LABELS below

We build a true-color-ish preview PNG from bands B04/B03/B02 (Red/Green/Blue)
so BLIP-VQA (a natural-image model) gets something visually sensible instead
of a raw 12-band array it can't consume.
"""
from __future__ import annotations

import csv
import os

import numpy as np

OUT_DIR = "data/bigearthnet_subset"
IMG_DIR = f"{OUT_DIR}/images"
LIMIT = 500  # plenty for a LoRA demo fine-tune; raise later if you have time/VRAM to spare

# Order matches the dataset card's "Class Labels" list exactly — required to
# turn the numeric `label` feature back into readable text.
CLASS_LABELS = [
    "Urban fabric", "Industrial or commercial units", "Arable land",
    "Permanent crops", "Pastures", "Complex cultivation patterns",
    "Land principally occupied by agriculture, with significant areas of natural vegetation",
    "Agro-forestry areas", "Broad-leaved forest", "Coniferous forest", "Mixed forest",
    "Natural grassland and sparsely vegetated areas",
    "Moors, heathland and sclerophyllous vegetation", "Transitional woodland, shrub",
    "Beaches, dunes, sands", "Inland wetlands", "Coastal wetlands",
    "Inland waters", "Marine waters",
]

# Sentinel-2 band order in this dataset's `optical_channel_wv` metadata is
# B01,B02,B03,B04,B05,B06,B07,B08,B8A,B09,B11,B12 (12 bands) — so
# Red=B04=index 3, Green=B03=index 2, Blue=B02=index 1.
_R, _G, _B = 3, 2, 1


def _percentile_stretch(band: np.ndarray) -> np.ndarray:
    band = band.astype("float32")
    lo, hi = np.percentile(band, [2, 98])
    if hi <= lo:
        lo, hi = float(band.min()), float(band.max() or 1.0)
    return np.clip((band - lo) / max(hi - lo, 1e-6), 0, 1)


def _optical_to_rgb_png(optical_array: np.ndarray):
    """optical_array expected shape: (bands, H, W) or (H, W, bands). Returns a PIL Image."""
    from PIL import Image

    arr = np.array(optical_array)
    if arr.ndim != 3:
        raise ValueError(f"Unexpected optical array shape {arr.shape} — inspect the raw feature and adjust this function.")

    # Normalize to (bands, H, W)
    if arr.shape[0] <= 13:          # bands first
        bands = arr
    elif arr.shape[-1] <= 13:       # bands last
        bands = np.moveaxis(arr, -1, 0)
    else:
        raise ValueError(f"Can't tell which axis is bands for shape {arr.shape} — print(arr.shape) and fix _R/_G/_B indices.")

    r = _percentile_stretch(bands[_R])
    g = _percentile_stretch(bands[_G])
    b = _percentile_stretch(bands[_B])
    rgb = (np.dstack([r, g, b]) * 255).astype("uint8")
    return Image.fromarray(rgb, mode="RGB")


def _labels_to_text(label_field) -> list[str]:
    """Handles either a multi-hot vector (length 19, 0/1) or a list of class indices."""
    arr = np.array(label_field)
    if arr.ndim == 1 and arr.size == len(CLASS_LABELS) and set(np.unique(arr)).issubset({0, 1}):
        idxs = np.nonzero(arr)[0]
    else:
        idxs = arr.astype(int).flatten()
    return [CLASS_LABELS[i] for i in idxs if 0 <= i < len(CLASS_LABELS)]


def main() -> None:
    from datasets import load_dataset

    os.makedirs(IMG_DIR, exist_ok=True)
    print("Opening GFM-Bench/BigEarthNet in streaming mode (no bulk download)...")
    ds = load_dataset(
        "GFM-Bench/BigEarthNet", split="train", streaming=True,
        trust_remote_code=True,  # this dataset uses an old-style loading script —
                                  # only works on datasets==2.21.0, see chat/README
    )

    rows = []
    for i, example in enumerate(ds):
        if i >= LIMIT:
            break

        if i == 0:
            print("First example keys:", list(example.keys()))
            print("optical type/shape:", type(example["optical"]), np.array(example["optical"]).shape)
            print("label raw value:", example["label"])

        try:
            img = _optical_to_rgb_png(example["optical"])
        except Exception as exc:  # noqa: BLE001
            print(f"Skipping row {i}: could not build RGB preview ({exc})")
            continue

        labels = _labels_to_text(example["label"])
        if not labels:
            continue

        fname = f"patch_{i:05d}.png"
        img.save(f"{IMG_DIR}/{fname}")
        rows.append({"image_path": fname, "labels": ";".join(labels)})

        if (i + 1) % 50 == 0:
            print(f"...{i + 1} patches processed")

    if not rows:
        raise SystemExit("No usable rows produced — check the printed shape/keys above and fix _optical_to_rgb_png / _labels_to_text.")

    manifest_path = f"{OUT_DIR}/manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "labels"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone. Saved {len(rows)} patches to {IMG_DIR}")
    print(f"Manifest written to {manifest_path}")
    print(
        "\nNext:\n"
        f"  python prepare_bigearthnet.py --manifest {manifest_path} "
        f"--image-root {IMG_DIR} --out data/bigearthnet_vqa_train.jsonl"
    )


if __name__ == "__main__":
    main()
