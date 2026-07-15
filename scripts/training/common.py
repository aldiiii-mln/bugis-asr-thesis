"""
Shared utilities across model training scripts.

Evaluation scheme per condition (Section 4.2):

    Condition       | Training data          | Eval to authentic test
    ----------------|-------------------------|----------------------------------
    Zero-Shot       | none                    | direct
    Authentic-Only  | 40 files (authentic)    | 5-fold CV (leakage risk, N small)
    Synthetic-Only  | 402 files (synthetic)   | direct on all 40 authentic files
    Combined        | 40 authentic + 402 synth| 5-fold CV (fold held out from
                     |                         | training each round)

Compute-matched (not epoch-matched) across conditions: total optimizer
steps should be equal, since data volume differs drastically between
conditions (e.g. MMS Authentic-Only at 60 steps vs Whisper at 200 steps in
early drafts was NOT compute-matched — this was a flagged issue, fixed here
by deriving steps from a fixed target rather than a fixed epoch count).
"""

from dataclasses import dataclass
from pathlib import Path
import random


@dataclass
class DataCondition:
    name: str  # "zero_shot" | "authentic_only" | "synthetic_only" | "combined"
    requires_kfold: bool
    train_files: list[str]


def make_kfold_splits(file_ids: list[str], k: int = 5, seed: int = 42) -> list[dict]:
    """Returns k splits, each {'train': [...], 'test': [...]}. For the
    Combined condition, pass only the authentic-derived file_ids here —
    synthetic files are added to every fold's training set separately
    (they carry no leakage risk against the held-out authentic test fold)."""
    rng = random.Random(seed)
    shuffled = file_ids.copy()
    rng.shuffle(shuffled)

    folds = [shuffled[i::k] for i in range(k)]
    splits = []
    for i in range(k):
        test = folds[i]
        train = [f for j, fold in enumerate(folds) if j != i for f in fold]
        splits.append({"train": train, "test": test})
    return splits


def compute_matched_steps(target_total_steps: int, dataset_size: int, batch_size: int) -> int:
    """Derive num_train_epochs (or max_steps directly, preferred) so that
    total optimizer steps match a fixed target across conditions with very
    different dataset sizes. Prefer passing max_steps directly to the
    Trainer rather than converting back to epochs, to avoid rounding
    drift."""
    steps_per_epoch = max(1, dataset_size // batch_size)
    epochs_needed = target_total_steps / steps_per_epoch
    return max(1, round(epochs_needed * steps_per_epoch))  # returns steps, not epochs


def verify_no_leakage(train_ids: list[str], test_ids: list[str]) -> bool:
    overlap = set(train_ids) & set(test_ids)
    if overlap:
        raise ValueError(f"Leakage detected: {overlap} present in both train and test")
    return True
