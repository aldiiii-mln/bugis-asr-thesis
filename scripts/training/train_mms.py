"""
MMS (facebook/mms-1b-all) fine-tuning for one data condition + fold.

Key design decision (methodology doc Section 4.1): warm-start from the
Indonesian ('ind') adapter, confirmed available in this checkpoint, rather
than random-init. Vocabulary is resized to accommodate the Bugis glottal
stop apostrophe, which the Indonesian adapter's vocab does not cover —
overlapping characters keep their pretrained embeddings, new characters
are randomly initialized.

All three MMS conditions (Authentic/Synthetic/Combined) are adapter-only
fine-tuning, so within-model comparison here is clean. The confound with
Whisper (full fine-tune vs adapter-only) only matters for cross-model
magnitude comparisons — see Claim 1b / Claim 2 in docs/methodology_summary.md.

Usage:
    python train_mms.py --condition combined --fold 0 \
        --data_dir ../../data --output_dir ./runs/mms-combined-f0
"""

import argparse
from pathlib import Path

import torch
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    TrainingArguments,
    Trainer,
)

from common import compute_matched_steps

MMS_CHECKPOINT = "facebook/mms-1b-all"
TARGET_TOTAL_STEPS = 200  # match Whisper's target for within-architecture-family comparability
BATCH_SIZE = 4  # MMS-1B is large; adjust for available VRAM


def load_condition_data(data_dir: Path, condition: str, fold: int):
    """TODO: same manifest contract as train_whisper.py — see that file."""
    raise NotImplementedError("Populate manifest loading logic")


def build_model_with_ind_adapter():
    processor = Wav2Vec2Processor.from_pretrained(MMS_CHECKPOINT, target_lang="ind")
    model = Wav2Vec2ForCTC.from_pretrained(MMS_CHECKPOINT, target_lang="ind", ignore_mismatched_sizes=True)
    model.load_adapter("ind")

    # TODO: resize vocabulary/output head to include Bugis glottal stop (')
    # and any characters not present in the 'ind' vocab. New rows should be
    # randomly initialized; overlapping characters retain pretrained weights.
    # See docs/methodology_summary.md Section 4.1 for the rationale.

    return model, processor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", required=True,
                         choices=["zero_shot", "authentic_only", "synthetic_only", "combined"])
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    model, processor = build_model_with_ind_adapter()

    if args.condition == "zero_shot":
        print("Zero-shot: skipping training, evaluate directly with 'ind' adapter as-is.")
        return

    train_ds, test_ds = load_condition_data(Path(args.data_dir), args.condition, args.fold)
    max_steps = compute_matched_steps(TARGET_TOTAL_STEPS, len(train_ds), BATCH_SIZE)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=BATCH_SIZE,
        max_steps=max_steps,
        save_strategy="steps",
        save_steps=max_steps,
        fp16=torch.cuda.is_available(),
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        tokenizer=processor.feature_extractor,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"Done. Trained for {max_steps} steps (target: {TARGET_TOTAL_STEPS}).")


if __name__ == "__main__":
    main()
