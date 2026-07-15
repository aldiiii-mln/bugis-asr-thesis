"""
Data quality control checks (Section 4.7 of the methodology doc). Run before
launching full experiments.

Checklist:
  1. Audio QC (authentic/GRN): noise/volume consistency, duration filter
     (<1s or >30s flagged).
  2. Audio QC (synthetic/TTS): manual spot-check by a Bugis speaker for
     unnatural pronunciation of Bugis-specific phonemes; synthesis
     artifacts (glitches, cut-off words).
  3. Transcript QC (blind re-transcription): inter-annotator agreement if
     2+ annotators available; otherwise a ~20% verification pass minimum.
  4. Text normalization: Unicode NFC normalization applied consistently to
     the WHOLE corpus (authentic + synthetic + Bible) BEFORE splitting data
     — run this first, not after.
  5. Duplication check: repeated sentences in the synthetic pool reduce
     effective linguistic diversity even when file count looks large
     (lesson from Common Voice Malta).
  6. Split/leakage check: verify no file appears in more than one
     fold/condition — this is the exact failure mode found in the
     LSTM/BiLSTM Bugis paper (Section 6 of methodology doc), where a small
     speaker pool without explicit speaker-independent splitting produced
     apparent memorization (WER 0% on multiple test rows).
"""

import hashlib
import unicodedata
from pathlib import Path
from collections import defaultdict


def check_duration_outliers(audio_dir: str, min_sec: float = 1.0, max_sec: float = 30.0):
    import soundfile as sf

    flagged = []
    for f in Path(audio_dir).glob("*.wav"):
        info = sf.info(str(f))
        duration = info.frames / info.samplerate
        if duration < min_sec or duration > max_sec:
            flagged.append((f.name, round(duration, 2)))
    return flagged


def normalize_nfc(text: str) -> str:
    """Apply BEFORE splitting data. Apply identically to authentic,
    synthetic, and Bible-derived text so visually-identical characters
    (e.g. apostrophes from different keyboards/sources) aren't counted as
    different characters during CER computation."""
    return unicodedata.normalize("NFC", text)


def check_duplicate_sentences(text_lines: list[str]) -> dict[str, list[int]]:
    """Returns {sentence_hash: [line_indices]} for any sentence appearing
    more than once. High duplication reduces effective linguistic
    diversity in the synthetic pool even if total file count is large."""
    seen = defaultdict(list)
    for i, line in enumerate(text_lines):
        key = hashlib.md5(normalize_nfc(line.strip()).encode("utf-8")).hexdigest()
        seen[key].append(i)
    return {k: v for k, v in seen.items() if len(v) > 1}


def check_fold_leakage(fold_assignments: dict[str, list[str]]) -> list[str]:
    """fold_assignments: {fold_name: [file_ids]}. Returns any file_id that
    appears in more than one fold — a hard sanity check to run once before
    the full experiment, given the leakage risk this exact QC gap produced
    in the LSTM/BiLSTM Bugis comparison paper."""
    file_to_folds = defaultdict(list)
    for fold_name, files in fold_assignments.items():
        for f in files:
            file_to_folds[f].append(fold_name)
    return [f for f, folds in file_to_folds.items() if len(folds) > 1]


if __name__ == "__main__":
    print("Import and call individual check functions from your preprocessing "
          "pipeline — this module has no standalone CLI by design, since each "
          "check needs paths specific to your local data layout.")
