# Data

Raw audio and text files are **not committed to this repository**. Each source
has its own licensing terms; keep raw data local (or in a private storage
bucket) and reference it via the paths below.

## `authentic_grn/`
Global Recordings Network (GRN), Bugis program #19080 ("Good News"). Free for
non-commercial/evangelism use; check current terms at globalrecordings.net
before redistributing any derived corpus.

- Expected contents: `audio/*.wav`, `transcripts/*.txt` (blind re-transcription,
  not the original GRN draft — see `docs/methodology_summary.md` on avoiding
  anchoring bias)

## `synthetic_tts/`
Edge TTS output reading domain-filtered NusaTranslation text (Bugis).
201 sampled paragraphs (seed=42) out of 606 domain-matched candidates,
filtered from 128,472 raw NusaTranslation sentences. See
`scripts/preprocessing/filter_domain_text.py` for the filtering pipeline.

- Expected contents: `audio/*.wav`, `source_text/paragraphs_201.txt`

## `bible_corpus/`
LAIBUG — Today's Bugis Version (1997, Lembaga Alkitab Indonesia / Indonesian
Bible Society). Audio ℗ 2023 Davar Partners International, distributed via
the Bible Brain API (Faith Comes By Hearing).

**Licensing note:** Davar audio is free to download and distribute for
non-commercial use only; the copyright notice must remain visible wherever
shared. Do not redistribute raw audio/text files in this public repo — keep
them local and gitignored. If releasing a derived training corpus publicly,
confirm terms with Faith Comes By Hearing / Indonesian Bible Society first.

**Open verification items before use (see docs/methodology_summary.md):**
- Confirm the Bugis dialect/variant used in this translation matches the
  GRN #19080 program's variant (Bugis has multiple regional variants —
  Bone, Wajo, Sinjai, Luwu, Barru, Sidrap, Soppeng, etc. — tracked separately
  by GRN/Ethnologue).
- The bundled SABDA-format PDF of this text has a broken font/encoding
  (missing 'é' and glottal-stop apostrophe throughout all 4,129 pages,
  confirmed against bible.com's LAIBUG text). **Do not use the PDF as a text
  source.** Use the Bible Brain API plain-text endpoint instead.
- Confirm whether this fileset has verse-level audio timing data available
  via Bible Brain (only a subset of bibleIds have it).
- Filter by book/genre to match GRN's narrative register (Gospels, Acts,
  Genesis-style narrative) before treating as domain-matched — the full
  Bible spans narrative, poetry, and epistle registers.

## `.gitignore` coverage

All audio files (`*.wav`, `*.mp3`), and anything under `data/*/audio/` or
`data/*/raw/`, are excluded from version control. Only small text manifests,
metadata, and documentation should be committed.
