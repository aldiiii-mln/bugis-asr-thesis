"""
Whisper fine-tuning (full fine-tune) for one data condition + fold.

Usage:
    python train_whisper.py --condition authentic_only --fold 0 \
        --model_size small --data_dir ../../data --output_dir ./runs/whisper-small-auth-f0
"""

import argparse
from pathlib import Path

import torch
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from datasets import Dataset, Audio

from common import make_kfold_splits, verify_no_leakage, compute_matched_steps


TARGET_TOTAL_STEPS = 200  # keep identical across all Whisper-model conditions
BATCH_SIZE = 8


def load_condition_data(data_dir: Path, condition: str, fold: int):
    """Build train/test HF Datasets for the given condition + fold index.
    TODO: wire up to actual manifest files once data/ is populated locally
    (this repo intentionally does not commit audio — see data/README.md)."""
    raise NotImplementedError(
        "Populate this with your actual manifest loading logic — "
        "expects columns: ['audio_path', 'text', 'file_id']"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", required=True,
                         choices=["zero_shot", "authentic_only", "synthetic_only", "combined"])
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--model_size", default="small", choices=["small", "medium"])
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    model_name = f"openai/whisper-{args.model_size}"
    processor = WhisperProcessor.from_pretrained(model_name, language="Indonesian", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(model_name)

    if args.condition == "zero_shot":
        print("Zero-shot: skipping training, evaluate directly.")
        return

    train_ds, test_ds = load_condition_data(Path(args.data_dir), args.condition, args.fold)

    max_steps = compute_matched_steps(TARGET_TOTAL_STEPS, len(train_ds), BATCH_SIZE)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=BATCH_SIZE,
        max_steps=max_steps,
        evaluation_strategy="no",  # evaluate separately via scripts/evaluation/
        save_strategy="steps",
        save_steps=max_steps,  # save final checkpoint only
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
        report_to=[],
    )

    trainer = Seq2SeqTrainer(
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
