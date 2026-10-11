"""
Convert a BigEarthNet.txt-style manifest into (image, question, answer)
VQA training triplets for backend/adaptation/train_lora_vqa.py.

Phase 5. Satisfies PS requirement: "remote-sensing fine-tuning or domain
adaptation using BigEarthNet.txt or any open source training data."

ASSUMPTION — stated explicitly because it matters:
The PS describes "BigEarthNet.txt" as containing co-registered
Sentinel-1/Sentinel-2 imagery plus text annotations (arxiv.org/abs/2603.29630).
This is NOT the same as the original public BigEarthNet benchmark, which
ships multi-label land-cover tags with no free-text captions. Until your
team has actually pulled and inspected the PS-linked dataset, do not
assume its manifest format matches what's below — VERIFY FIRST.

This script expects a manifest CSV/JSONL with, at minimum:
    image_path,label_1;label_2;...        (multi-label land-cover tags)
or
    image_path,caption                    (free-text description)

and converts each row into simple templated VQA pairs. Templating is a
deliberate hackathon-scope choice — it is not attempting to reproduce
whatever generation pipeline the PS dataset's authors used, and every
converted example should be understood as "training signal for basic
land-cover VQA", not "ground-truth human annotation".

Usage:
    python prepare_bigearthnet.py \
        --manifest /path/to/bigearthnet_manifest.csv \
        --image-root /path/to/bigearthnet/images \
        --out data/bigearthnet_vqa_train.jsonl \
        --limit 2000   # optional cap — keep Day-3 fine-tuning runs short
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

random.seed(42)

# Templated question variants per example so the model doesn't only ever
# see one exact question phrasing. Kept small and generic on purpose.
_LABEL_QUESTIONS = [
    "What land cover types are present in this image?",
    "Describe the land cover visible in this satellite image.",
    "What does this satellite image show?",
]
_CAPTION_QUESTIONS = [
    "Describe this satellite image.",
    "What does this image show?",
    "Give a brief description of this scene.",
]


def _read_manifest(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        if path.suffix == ".jsonl":
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
        else:
            reader = csv.DictReader(f)
            yield from reader


def build_examples(manifest_path: Path, image_root: Path, limit: int | None) -> list[dict]:
    examples: list[dict] = []
    for row in _read_manifest(manifest_path):
        image_path = str(image_root / row["image_path"])

        if row.get("caption"):
            question = random.choice(_CAPTION_QUESTIONS)
            answer = row["caption"].strip()
        elif row.get("label_1") or row.get("labels"):
            labels_raw = row.get("labels") or ";".join(
                v for k, v in row.items() if k.startswith("label_") and v
            )
            labels = [l.strip() for l in labels_raw.split(";") if l.strip()]
            if not labels:
                continue
            question = random.choice(_LABEL_QUESTIONS)
            answer = ", ".join(labels)
        else:
            continue  # row has neither a caption nor labels — skip, don't guess

        examples.append({"image_path": image_path, "question": question, "answer": answer})
        if limit and len(examples) >= limit:
            break

    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    examples = build_examples(args.manifest, args.image_root, args.limit)
    if not examples:
        raise SystemExit(
            "No usable rows found. Check that the manifest has a 'caption' or "
            "'labels'/'label_1' column — see this file's docstring."
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    print(f"Wrote {len(examples)} VQA training examples to {args.out}")


if __name__ == "__main__":
    main()
