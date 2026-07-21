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


# PENTING: WER dan CER wajib pakai langkah normalisasi yang SAMA PERSIS
# (lowercase, rapikan spasi, strip) — cuma beda di reduksi akhir (kata vs
# karakter), karena itu memang perbedaan definisi WER/CER itu sendiri, bukan
# perbedaan normalisasi. Kalau tidak disamakan, gap CER-vs-WER antar kondisi
# (dasar Klaim 3) bisa bias oleh hal remeh kayak beda kapitalisasi, bukan
# oleh perbedaan ortografi yang sebenarnya mau diukur.
WER_TRANSFORM = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfWords(),
])

CER_TRANSFORM = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfChars(),
])


def normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def compute_wer(references: list[str], hypotheses: list[str]) -> float:
    references = [normalize(r) for r in references]
    hypotheses = [normalize(h) for h in hypotheses]
    # jiwer >=4.0 mengganti nama kwarg truth_transform -> reference_transform
    return jiwer.wer(references, hypotheses, reference_transform=WER_TRANSFORM, hypothesis_transform=WER_TRANSFORM)


def compute_cer(references: list[str], hypotheses: list[str]) -> float:
    references = [normalize(r) for r in references]
    hypotheses = [normalize(h) for h in hypotheses]
    return jiwer.cer(references, hypotheses, reference_transform=CER_TRANSFORM, hypothesis_transform=CER_TRANSFORM)


def compute_both(references: list[str], hypotheses: list[str]) -> dict:
    return {
        "wer": compute_wer(references, hypotheses),
        "cer": compute_cer(references, hypotheses),
        "n_samples": len(references),
    }


def evaluate_manifest(manifest_path: str) -> dict:
    """manifest: CSV dengan kolom file_id, reference, hypothesis — format yang
    sama dengan output evaluate_model() di zero_shot_eval_*.py."""
    import csv
    refs, hyps = [], []
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            refs.append(row["reference"])
            hyps.append(row["hypothesis"])
    return compute_both(refs, hyps)


# ---- RUN (gaya Colab, tanpa argparse — lihat metodologi 9.7) ----
if __name__ == "__main__":
    import json

    # ---- EDIT BAGIAN INI ----
    MANIFEST_PATH = "/content/zero_shot_results/zeroshot_whisper-small_predictions.csv"
    # --------------------------

    results = evaluate_manifest(MANIFEST_PATH)
    print(json.dumps(results, indent=2))
