# Remote-sensing adaptation pipeline (Phase 5)

Satisfies the PS-mandatory requirement: *"At least one visual or
vision-language component must be fine-tuned or otherwise adapted using
BigEarthNet.txt or the any open source training data."*

## Pipeline

```
BigEarthNet.txt manifest ──▶ prepare_bigearthnet.py ──▶ *_vqa_train.jsonl
                                                                │
                                                                ▼
                                                     train_lora_vqa.py
                                                                │
                                                                ▼
                              backend/models/checkpoints/vqa_lora_bigearthnet/
                                                                │
                                                                ▼
                                        backend/models/vqa.py loads it
                                        automatically on next restart
```

## Status: scripts are real and runnable — no adapter is checked in

`backend/models/checkpoints/vqa_lora_bigearthnet/` is empty in this
repo. Until someone actually runs `train_lora_vqa.py`, the backend
correctly reports `adapted: false` in every VQA response and uses base
`Salesforce/blip-vqa-base` weights. **This is the honest state, not a
bug** — do not check in a fabricated/untrained adapter directory just to
flip that flag.

## Before running this for real

1. **Get the actual PS-linked dataset** (arxiv.org/abs/2603.29630) and
   inspect its real manifest format. `prepare_bigearthnet.py`'s expected
   CSV/JSONL columns (`caption` or `labels`/`label_1..n`) are a
   reasonable guess based on the PS description, not a verified schema
   — update the parsing in that file to match what you actually
   download before trusting its output.
2. Decide compute: LoRA fine-tuning of BLIP-VQA-base is realistic on a
   single consumer GPU in well under a day for a few thousand examples;
   CPU-only will be much slower — cap `--limit` accordingly in
   `prepare_bigearthnet.py`.
3. Install training-only deps (`pip install peft accelerate datasets`)
   — these are deliberately NOT in `backend/requirements.txt` since the
   running API doesn't need `datasets`/`accelerate`, only `peft` at
   inference time to load the resulting adapter.

## Phase 12 — what was actually verified this session (and what wasn't)

Attempted, in the sandbox used to build Phase 12: install `torch`/`transformers`/`peft`
and actually run this pipeline end-to-end. Result, stated plainly:

- **`prepare_bigearthnet.py` was smoke-tested for real** against a small hand-built,
  clearly-synthetic manifest (5 rows, random-noise PNGs, invented captions/labels —
  nothing from the actual BigEarthNet.txt dataset). It ran cleanly and produced a
  correctly-shaped `vqa_train.jsonl`. This proves the CSV/JSONL parsing logic and
  the templated-question generation work as written — it does **not** prove
  `prepare_bigearthnet.py`'s column-name assumptions (`caption` / `labels` /
  `label_1..n`) match the real PS-linked dataset's actual schema, because that
  dataset was never fetched (see below). Verify that for real before trusting this
  script's output on the real data.
- **`train_lora_vqa.py` was NOT run**, even on the synthetic data above, and could
  not be in this environment: (1) the dataset host (arxiv.org and whatever hosts
  the actual BigEarthNet.txt files) is not reachable from the sandbox this Phase
  was built in, and (2) `huggingface.co` — needed to download the base
  `Salesforce/blip-vqa-base` weights — is also not reachable from it. Both are
  infrastructure constraints of that particular build environment, not a flaw in
  the script. `backend/models/checkpoints/vqa_lora_bigearthnet/` remains empty;
  `adapted: false` remains the honest, correct state.

**Bottom line, unchanged from every prior phase:** running this for real — with the
actual PS-linked dataset and a machine that can reach huggingface.co — is still an
execution task for whoever has that access, not something that can be faked here.
Do not check in a checkpoint that wasn't actually produced by a real training run.
