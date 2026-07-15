# Automatic Speech Recognition for the Low-Resource Bugis Language

**Evaluating Authentic-Synthetic Data Combination on Whisper, OWSM, and MMS**

Undergraduate thesis project (Universiti Teknikal Malaysia Melaka / UTeM) studying
Automatic Speech Recognition for Bugis — a low-resource Austronesian language spoken
in South Sulawesi, Indonesia.

## Overview

This project fine-tunes and evaluates three ASR architectures (Whisper, a second
model — OWSM or Whisper-medium, and MMS) under three data conditions
(Authentic-Only, Synthetic-Only, Combined) to test whether combining authentic
speech with synthetic (TTS) data improves generalization to real Bugis speech,
compared to either data source alone.

The central claim under test: does combining authentic and synthetic data improve
generalization to authentic Bugis speech, and does this pattern replicate
independently across architectures (within-model comparison, not a claim of
strict cross-architecture validation)?

## Data Sources

| Source | Role | Notes |
|---|---|---|
| Global Recordings Network (GRN), Bugis program #19080 | Authentic-Only training + test | ~40 files, ~44 min, 1 speaker |
| Edge TTS (Indonesian voice reading Bugis text, via NusaTranslation) | Synthetic-Only training | ~402 files, ~131 min, domain-filtered for religious/narrative register |
| LAIBUG — Today's Bugis Version (1997, Indonesian Bible Society), audio by Davar Partners International (2023) | Authentic corpus expansion (in evaluation) | Accessed via the Bible Brain API (Faith Comes By Hearing); multiple narrators |

Raw audio and text data are **not committed to this repository** — see
`data/README.md` for licensing constraints and how to obtain each source.

## Repository Structure

```
bugis-asr-thesis/
├── data/                  # Data source notes and licensing info (raw files gitignored)
├── scripts/
│   ├── preprocessing/     # Domain filtering, audio-text alignment, QC
│   ├── training/          # Fine-tuning scripts per model
│   └── evaluation/        # WER/CER computation, bootstrap CI, k-fold
├── notebooks/             # Exploratory analysis
└── docs/                  # Methodology notes
```

## Methodology Summary

See `docs/methodology_summary.md` for the full experimental design: evaluation
scheme per condition, k-fold cross-validation for leakage-prone conditions,
compute-matched (not epoch-matched) training, blind re-transcription for ground
truth, and WER/CER reported together to support the orthographic-inconsistency
analysis.

## Status

In progress — thesis research, non-commercial academic use only.

## Author

Muhammad Aldi Maulana
