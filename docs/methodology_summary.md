# Methodology Summary

Condensed reference. For the full derivation (literature review, rejected
alternatives, iteration history), see the thesis document itself — this
file exists so the repo is self-explanatory without it.

## Claims

**Claim 1a (Methodological gap).** A prior Bugis speech-to-text study
(LSTM vs BiLSTM, IOTA/ASCEE journal, Sinta 5) exists, so "no baseline"
cannot be claimed. Its results (BiLSTM WER 4.88%, LSTM WER 90.33%) show
signs of speaker-specific memorization (5 speakers, no explicit
speaker-independent split, multiple exact-match predictions) and confound
bidirectionality with a doubling of hidden units (128 vs 256). This thesis
addresses those specific gaps: k-fold CV, single-variable within-model
comparison, blind re-transcription, modern pretrained architectures.

**Claim 1b (Population gap — main claim).** Combining authentic and
synthetic data for fine-tuning improves generalization to real Bugis
speech versus either source alone. Tested independently across three
models (within-model comparison — not a claim of validated
cross-architecture universality). Whether the pattern replicates across
all three models is reported descriptively, not as proof of a universal
law.

**Claim 2 (Causal-attribution caution).** If MMS doesn't replicate the
Claim 1b pattern, non-replication is reported honestly — without claiming
CTC architecture itself is the cause, since MMS's adapter-only fine-tuning
vs Whisper/OWSM's full fine-tuning remains an unresolved confound for any
causal claim.

**Claim 3 (Orthographic inconsistency).** Some WER reflects unstandardized
Bugis spelling, not genuine transcription error — supported quantitatively
by CER reported alongside WER (see `scripts/evaluation/compute_wer_cer.py`).

**What NOT to claim:** universal cross-architecture validation; "ready for
real-world use"; synthetic data equivalent to authentic speech; CTC
inherently weaker than encoder-decoder; statistical significance without
running the actual CI/test.

## Experimental Design

3 models × 3 data conditions:
- **Models:** Whisper-small (full fine-tune), a second model (Whisper-medium
  or OWSM-small — see open items), MMS (adapter-only, `ind` warm-start)
- **Conditions:** Authentic-Only (40 GRN files), Synthetic-Only (402 Edge
  TTS files), Combined

Evaluation scheme, compute-matching, and QC checklist: see
`scripts/training/common.py` and `scripts/preprocessing/qc_checks.py`
docstrings — kept in code so they can't drift out of sync with what's
actually implemented.

## Known Limitations (to state explicitly in the thesis)

- K-fold CV here tests generalization to unseen *utterances* from the
  same speaker, not to unseen speakers — the authentic corpus has only
  one GRN speaker.
- Multi-seed evaluation is applied only where tractable: not for
  Authentic-Only (40 files, extreme scarcity), tractable in principle for
  Synthetic-Only and Combined.
- NusaTranslation source text is Indonesian→Bugis translation, not
  organic Bugis writing — "translationese" risk remains even after
  domain filtering.

## Open Items (as of last update)

- [ ] Confirm Bugis dialect/variant match between GRN #19080 and the
  LAIBUG Bible translation before treating them as the same "authentic
  Bugis" pool (Bugis has multiple tracked regional variants).
- [ ] Second model decision: Whisper-medium (safe) vs OWSM-small (more
  architecturally distinct, more setup friction via ESPnet).
- [ ] Consistency pass: replace "Cross-Architecture Validation" phrasing
  throughout the draft with the revised title's framing.
- [ ] Full author details for the IOTA/ASCEE and Makassar (UMI) papers,
  for citation.
- [ ] Bible Brain API access: confirm whether the LAIBUG fileset has
  verse-level audio timing; if not, use an independent forced-aligner
  (not Whisper) — see `scripts/preprocessing/align_audio_text.py`.
