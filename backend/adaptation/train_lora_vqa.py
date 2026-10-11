"""
LoRA fine-tuning: adapt Salesforce/blip-vqa-base to remote-sensing
imagery using the BigEarthNet-derived VQA triplets from
prepare_bigearthnet.py.

Phase 5 (P0). This IS the PS-mandatory "remote-sensing fine-tuning or
domain adaptation" component. It is intentionally lightweight (LoRA,
not full fine-tuning) so it fits the 4-day budget and runs on a single
consumer GPU (or CPU, slowly) — full fine-tuning of even a 385M-param
model on limited hackathon compute/time is not a realistic ask.

This is a genuine training script, not a placeholder — run it for real
before the demo, even for a small number of steps on a subset of data.
A LoRA adapter trained for 20 minutes on a few hundred examples is an
honest "adapted" claim. Skipping this file entirely and shipping the
base model is also honest, AS LONG AS the app reports adapted=False
(it does, by default — see models/vqa.py).

Phase 14 addition — "best model" checkpointing:
Earlier version of this script trained for N epochs and saved
whatever the LAST epoch produced, with no validation signal at all —
if training overfit or got worse in a later epoch, you'd have no way
to know or recover the better earlier checkpoint. This version:
  1. Holds out a validation split (auto-split from --train-jsonl if
     --val-jsonl isn't given).
  2. Evaluates on that split every epoch.
  3. Uses HF Trainer's load_best_model_at_end so the checkpoint that
     is actually saved to --output-dir is the one with the LOWEST
     validation loss seen during training, not simply the final epoch.
  4. Writes adapter_info.json next to the saved adapter recording the
     real numbers (train/val example counts, per-epoch eval losses,
     best eval loss, which epoch it came from) — no invented metrics,
     everything in that file is read straight from the Trainer's own
     log history.

Usage:
    python train_lora_vqa.py \
        --train-jsonl data/bigearthnet_vqa_train.jsonl \
        --output-dir ../models/checkpoints/vqa_lora_bigearthnet \
        --epochs 3 --batch-size 4

    # or, with an explicit held-out validation file instead of an
    # auto-split of --train-jsonl:
    python train_lora_vqa.py \
        --train-jsonl data/bigearthnet_vqa_train.jsonl \
        --val-jsonl data/bigearthnet_vqa_val.jsonl \
        --output-dir ../models/checkpoints/vqa_lora_bigearthnet \
        --epochs 3 --batch-size 4

Dependencies (not in the main backend requirements.txt — training-only):
    pip install peft accelerate datasets
"""
from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset


class VQAJsonlDataset(Dataset):
    def __init__(self, examples: list[dict], processor):
        self.examples = examples
        self.processor = processor

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        image = Image.open(ex["image_path"]).convert("RGB")
        encoding = self.processor(
            image, ex["question"], text_target=ex["answer"],
            return_tensors="pt", padding="max_length", truncation=True, max_length=32,
        )
        return {k: v.squeeze(0) for k, v in encoding.items()}


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _split_train_val(examples: list[dict], val_split: float, seed: int = 42) -> tuple[list[dict], list[dict]]:
    """Deterministic shuffle + split so re-running with the same
    --train-jsonl and --val-split always produces the same split
    (reproducible, not re-randomized every run)."""
    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_split)) if len(shuffled) > 1 else 0
    return shuffled[n_val:], shuffled[:n_val]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", required=True, type=Path)
    parser.add_argument(
        "--val-jsonl", type=Path, default=None,
        help="Optional separate validation set. If omitted, a slice of --train-jsonl is held out instead.",
    )
    parser.add_argument(
        "--val-split", type=float, default=0.1,
        help="Fraction of --train-jsonl held out for validation when --val-jsonl is not given.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--base-model", default="Salesforce/blip-vqa-base")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-r", type=int, default=8)
    args = parser.parse_args()

    from peft import LoraConfig, get_peft_model
    from transformers import BlipForQuestionAnswering, BlipProcessor, Trainer, TrainingArguments

    processor = BlipProcessor.from_pretrained(args.base_model)
    model = BlipForQuestionAnswering.from_pretrained(args.base_model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        # BLIP's text decoder/vision encoder use standard attention linear
        # layers named query/key/value — target those. If this errors on
        # your installed transformers version, run
        # `print(model)` first and adjust target_modules accordingly.
        target_modules=["query", "value"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    all_train_examples = _load_jsonl(args.train_jsonl)
    if args.val_jsonl is not None:
        train_examples = all_train_examples
        val_examples = _load_jsonl(args.val_jsonl)
    else:
        train_examples, val_examples = _split_train_val(all_train_examples, args.val_split)

    if len(val_examples) == 0:
        raise SystemExit(
            f"No validation examples available (train set has {len(all_train_examples)} rows, "
            f"--val-split={args.val_split}). Provide more training data, a lower --val-split "
            "denominator won't help below 1 example, or pass --val-jsonl explicitly. Training "
            "without ANY validation signal means there is no way to pick a 'best' checkpoint — "
            "refusing to silently fall back to 'just save the last epoch'."
        )

    print(f"Train examples: {len(train_examples)} | Validation examples: {len(val_examples)}")

    train_dataset = VQAJsonlDataset(train_examples, processor)
    val_dataset = VQAJsonlDataset(val_examples, processor)

    eval_batch_size = args.eval_batch_size or args.batch_size

    training_args = TrainingArguments(
        output_dir=str(args.output_dir / "_trainer_tmp"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=eval_batch_size,
        learning_rate=args.lr,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,          # keep disk usage sane — only best + most-recent checkpoint dirs
        load_best_model_at_end=True, # <- the actual "best model" mechanism
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=[],
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    trainer.train()

    # trainer.model now holds the BEST checkpoint by eval_loss (thanks to
    # load_best_model_at_end=True above), not necessarily the final epoch.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.output_dir))

    # Pull the real per-epoch eval losses straight out of the Trainer's own
    # log history — nothing here is invented or estimated.
    eval_logs = [entry for entry in trainer.state.log_history if "eval_loss" in entry]
    best_entry = min(eval_logs, key=lambda e: e["eval_loss"]) if eval_logs else None

    info = {
        "base_model": args.base_model,
        "adaptation_method": "LoRA (peft)",
        "lora_r": args.lora_r,
        "epochs_requested": args.epochs,
        "train_examples": len(train_examples),
        "val_examples": len(val_examples),
        "val_source": "explicit --val-jsonl" if args.val_jsonl is not None else f"auto-split ({args.val_split:.0%} of --train-jsonl)",
        "per_epoch_eval_loss": [
            {"epoch": e.get("epoch"), "eval_loss": e.get("eval_loss")} for e in eval_logs
        ],
        "best_eval_loss": best_entry["eval_loss"] if best_entry else None,
        "best_epoch": best_entry.get("epoch") if best_entry else None,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "note": (
            "best_eval_loss/best_epoch identify which checkpoint load_best_model_at_end "
            "selected and saved to this directory — it is not necessarily the final epoch. "
            "This is a validation-set loss (next-token cross-entropy), not a task-accuracy "
            "benchmark figure; do not present it to judges as validated VQA accuracy."
        ),
    }
    (args.output_dir / "adapter_info.json").write_text(json.dumps(info, indent=2))

    print(f"Saved BEST LoRA adapter (by eval_loss) to {args.output_dir}")
    if best_entry:
        print(f"Best eval_loss: {best_entry['eval_loss']:.4f} (epoch {best_entry.get('epoch')})")
    print(f"Full per-epoch metrics written to {args.output_dir / 'adapter_info.json'}")
    print(
        "Set VQA_LORA_ADAPTER_PATH to this directory (or leave the default "
        "in models/vqa.py, which already points here) so the backend picks "
        "it up on next restart."
    )


if __name__ == "__main__":
    main()

