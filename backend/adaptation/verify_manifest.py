"""
Offline schema check for a BigEarthNet.txt-style manifest, BEFORE running
prepare_bigearthnet.py against it.

Phase 15. Closes part of the "adaptation debt" flagged since Phase 5/12:
adaptation/README.md has said since Phase 5 that prepare_bigearthnet.py's
column-name assumptions (`caption` / `labels` / `label_1..n`) are "a
reasonable guess based on the PS description, not a verified schema" and
that whoever runs the pipeline for real must "update the parsing in that
file to match what you actually download". This script is that check,
built in advance so it's a five-second command instead of a manual
inspection once the real dataset is finally reachable.

WHAT THIS DOES NOT DO (read before assuming more happened):
- It does NOT download BigEarthNet.txt. This sandbox's network egress is
  restricted to package registries only (pypi/npm/github release assets)
  — it cannot reach arxiv.org or any dataset host, so the real manifest
  was never available to test against here. See PHASE15.md / LEFT.md.
- It does NOT run any model or training step. It only parses a manifest
  file already sitting on disk and reports whether prepare_bigearthnet.py
  would find anything usable in it.
- It does NOT modify prepare_bigearthnet.py's parsing logic. If this
  script finds the real manifest's columns don't match, the fix is to
  edit build_examples() in prepare_bigearthnet.py — this script's job is
  only to tell you that BEFORE you burn a training run finding out.

Usage:
    python verify_manifest.py --manifest /path/to/downloaded/manifest.csv

Exit code is 0 if at least one row looks usable, 1 otherwise — safe to
wire into a shell script/CI step ahead of the real training run.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


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


def check(manifest_path: Path, sample_size: int = 5) -> dict:
    rows = list(_read_manifest(manifest_path))
    total = len(rows)

    has_image_path = sum(1 for r in rows if r.get("image_path"))
    has_caption = sum(1 for r in rows if r.get("caption"))
    has_labels = sum(
        1 for r in rows if r.get("labels") or any(k.startswith("label_") and v for k, v in r.items())
    )
    usable = sum(
        1
        for r in rows
        if r.get("image_path") and (r.get("caption") or r.get("labels") or any(k.startswith("label_") and v for k, v in r.items()))
    )

    columns_seen = sorted({k for r in rows for k in r.keys()}) if rows else []
    sample = rows[:sample_size]

    return {
        "manifest": str(manifest_path),
        "total_rows": total,
        "rows_with_image_path": has_image_path,
        "rows_with_caption": has_caption,
        "rows_with_labels": has_labels,
        "usable_rows": usable,
        "usable_pct": round(100 * usable / total, 1) if total else 0.0,
        "columns_seen": columns_seen,
        "sample_rows": sample,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--sample-size", type=int, default=5)
    args = parser.parse_args()

    if not args.manifest.exists():
        print(f"ERROR: manifest not found at {args.manifest}", file=sys.stderr)
        raise SystemExit(1)

    report = check(args.manifest, args.sample_size)

    print("BigEarthNet.txt manifest check")
    print("=" * 40)
    print(f"Manifest:          {report['manifest']}")
    print(f"Total rows:        {report['total_rows']}")
    print(f"Columns seen:      {report['columns_seen']}")
    print(f"Have image_path:   {report['rows_with_image_path']}")
    print(f"Have caption:      {report['rows_with_caption']}")
    print(f"Have labels:       {report['rows_with_labels']}")
    print(f"USABLE rows:       {report['usable_rows']} ({report['usable_pct']}%)")
    print()

    if report["usable_rows"] == 0:
        print(
            "RESULT: FAIL — prepare_bigearthnet.py would produce zero training examples "
            "from this file. Its build_examples() expects an 'image_path' column plus "
            "either a 'caption' column or 'labels'/'label_1..n' columns (see that file's "
            "docstring). Inspect columns_seen above against what this manifest actually "
            "has, and update build_examples() to match before running training.",
            file=sys.stderr,
        )
        print(f"Sample rows for inspection: {json.dumps(report['sample_rows'], indent=2)}", file=sys.stderr)
        raise SystemExit(1)

    if report["usable_pct"] < 50:
        print(
            f"RESULT: PARTIAL — only {report['usable_pct']}% of rows are usable as-is. "
            "prepare_bigearthnet.py will silently skip the rest (by design — it does not "
            "guess). Check columns_seen for a column name mismatch before assuming this is "
            "just missing data in the source dataset."
        )
    else:
        print("RESULT: OK — most rows look usable. Safe to run prepare_bigearthnet.py against this manifest.")


if __name__ == "__main__":
    main()
