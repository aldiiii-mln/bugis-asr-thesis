"""
Domain-matching filter for NusaTranslation source text (Section 4.8 of the
thesis methodology). Filters raw NusaTranslation sentences to match GRN's
oral-narrative, non-systematic-theology register, via whole-sentence
exclusion (not partial cleaning) for structurally mismatched patterns.

Pipeline (9 iterative rounds, each triggered by a manual spot-check finding
— see docs/methodology_summary.md Section 4.8 for the full derivation):

  1. Christian-topic keyword match + Islamic-term exclusion
  2. Encyclopedic-style heuristics (repeated "bahasa X", year/century patterns)
  3. Fix: um­ma' exclusion narrowed to the phrase "umma' muhammad" (word-level
     exclusion was wrongly discarding valid Christian sentences)
  4. Decision: switch from partial-line cleaning to whole-line exclusion
  5. Whole-line exclusion: numerals, parentheses, colons, isolated chars,
     Bible-reference citations, CSV artifacts, wiki/HTML markup, URLs,
     control/zero-width chars, repeated symbols, bracketed citations,
     backslash-escapes
  6. Add: domain-web + academic/journalistic markers (e.g. "sejarawan",
     "media sosial") after finding historian citations and news references
     slipping through
  7. Fix: 'é' detection generalized to match anywhere in the word (not just
     standalone). Add: exclamation marks, spaced commas.
     NOTE: hyphen exclusion was tried here and found to discard 72% of data
     (923/1284) because Bugis reduplication (e.g. "rasul-rasulna") uses
     hyphens — hyphen exclusion was DISABLED after this finding.
  8. Add: em-dash / en-dash exclusion, separate from hyphen (safe, since
     Bugis reduplication uses plain hyphens, not em/en-dashes)
  9. Direct-speech quotation marks: STRIPPED (character removed, line kept),
     not excluded — GRN's narrative style includes dialogue without formal
     quotation punctuation.

Verified reproducible end-to-end run (14 Jul 2026): 128,472 -> 3,863
(keyword+heuristic stage) -> 606 (after all whole-line exclusion rounds)
-> 201 sampled paragraphs (seed=42).

LIMITATION (documented, not chased further): named-entity / citation-style
content (historical figures, broadcast media references) is only partially
caught by keyword matching and would require NER or manual verification to
fully resolve. See docs/methodology_summary.md for the exact limitation
text used in the thesis.
"""

import re
import random
from pathlib import Path

SEED = 42
SAMPLE_SIZE = 201

# TODO: fill in with the exact keyword lists used in your verified run.
# Keeping these as named, editable constants (rather than inline) so the
# filtering logic stays auditable against Section 4.8 of the methodology doc.
CHRISTIAN_KEYWORDS = [
    # e.g. "yesus", "kristus", "injil", ...
]
ISLAMIC_EXCLUSION_TERMS = [
    # e.g. "nabi muhammad", "al-quran", ... (excluding "umma'" alone —
    # narrowed to "umma' muhammad" per round 3 fix)
]

WHOLE_LINE_EXCLUSION_PATTERNS = [
    r"\d",                          # numerals
    r"[()]",                        # parentheses
    r":",                           # colons
    r"[\u2013\u2014]",              # em-dash / en-dash (NOT plain hyphen -)
    r"https?://\S+",                # URLs
    r"<[^>]+>",                     # HTML/wiki markup
    r"\[\d+\]",                     # bracketed citations
    r"\\",                          # backslash escapes
    # TODO: add Bible-reference pattern, CSV-artifact pattern,
    # academic/journalistic marker list, domain-web marker list —
    # transcribe from your verified run for exact reproducibility.
]

QUOTE_STRIP_CHARS = ['"', "\u201c", "\u201d"]  # stripped, not exclusionary


def is_encyclopedic(sentence: str) -> bool:
    """Heuristic from round 2: repeated 'bahasa X' pattern, year/century refs."""
    if len(re.findall(r"\bbahasa\s+\w+", sentence, flags=re.IGNORECASE)) >= 2:
        return True
    if re.search(r"\b(abad|tahun)\s+\d", sentence, flags=re.IGNORECASE):
        return True
    return False


def matches_topic(sentence: str) -> bool:
    lower = sentence.lower()
    if any(term in lower for term in ISLAMIC_EXCLUSION_TERMS):
        return False
    return any(kw in lower for kw in CHRISTIAN_KEYWORDS)


def should_exclude_whole_line(sentence: str) -> bool:
    return any(re.search(pat, sentence) for pat in WHOLE_LINE_EXCLUSION_PATTERNS)


def strip_quotes(sentence: str) -> str:
    for ch in QUOTE_STRIP_CHARS:
        sentence = sentence.replace(ch, "")
    return sentence


def filter_corpus(raw_sentences: list[str]) -> list[str]:
    stage1 = [s for s in raw_sentences if matches_topic(s) and not is_encyclopedic(s)]
    stage2 = [strip_quotes(s) for s in stage1 if not should_exclude_whole_line(s)]
    return stage2


def main(input_path: str, output_path: str):
    raw = Path(input_path).read_text(encoding="utf-8").splitlines()
    filtered = filter_corpus(raw)

    print(f"Raw: {len(raw)} -> Filtered: {len(filtered)}")

    random.seed(SEED)
    sample = random.sample(filtered, min(SAMPLE_SIZE, len(filtered)))

    Path(output_path).write_text("\n".join(sample), encoding="utf-8")
    print(f"Sampled {len(sample)} paragraphs (seed={SEED}) -> {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Raw NusaTranslation text, one sentence per line")
    parser.add_argument("--output", required=True, help="Output path for sampled paragraphs")
    args = parser.parse_args()
    main(args.input, args.output)
