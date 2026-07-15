"""
WER + CER computation, reported side by side (Section 4.3).

CER supports Claim 3 (orthographic-inconsistency hypothesis): if a
meaningful share of WER reflects unstandardized Bugis spelling rather than
genuine transcription error, the gap between conditions should be smaller
in CER than in WER, since character-level slips (e.g. a missing glottal
apostrophe) don't get penalized as heavily as a full wrong-word count.

Both metrics MUST use identical text normalization (NFC, glottal apostrophe
preserved) — see scripts/preprocessing/qc_checks.py:normalize_nfc.
"""

import jiwer
from pathlib import Path
import unicodedata


TRANSFORM = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfWords(),
])


def normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def compute_wer(references: list[str], hypotheses: list[str]) -> float:
    references = [normalize(r) for r in references]
    hypotheses = [normalize(h) for h in hypotheses]
    return jiwer.wer(references, hypotheses, truth_transform=TRANSFORM, hypothesis_transform=TRANSFORM)


def compute_cer(references: list[str], hypotheses: list[str]) -> float:
    references = [normalize(r) for r in references]
    hypotheses = [normalize(h) for h in hypotheses]
    return jiwer.cer(references, hypotheses)


def compute_both(references: list[str], hypotheses: list[str]) -> dict:
    return {
        "wer": compute_wer(references, hypotheses),
        "cer": compute_cer(references, hypotheses),
        "n_samples": len(references),
    }


def evaluate_manifest(manifest_path: str) -> dict:
    """manifest: tab-separated file_id\\treference\\thypothesis per line."""
    refs, hyps = [], []
    for line in Path(manifest_path).read_text(encoding="utf-8").splitlines():
        _, ref, hyp = line.split("\t")
        refs.append(ref)
        hyps.append(hyp)
    return compute_both(refs, hyps)


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    results = evaluate_manifest(args.manifest)
    print(json.dumps(results, indent=2))
