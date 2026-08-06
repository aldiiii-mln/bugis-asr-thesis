"""
evaluate_synthetic_only_whisper.py

Tujuan
------
Menutup loop evaluasi untuk Whisper-small di kondisi Synthetic-Only, dengan
REUSE LANGSUNG compute_wer_cer.py + bootstrap_ci.py (import, BUKAN disalin
ulang) -- pola SAMA PERSIS dengan evaluate_authentic_only_whisper.py.

Beda dari evaluate_authentic_only_whisper.py: HANYA titik masuk (path CSV
prediksi) yang beda -- logika perhitungannya (bootstrap_ci group-aware per
rekaman, load_recording_mapping, resolve_groups, print_grouping_diagnostic)
TIDAK BERUBAH SAMA SEKALI. Ini karena bootstrap_ci.py sudah generik terhadap
skema training (fold atau bukan) -- yang penting cuma format CSV prediksi
(file_id, reference, hypothesis) dan RECORDING_MAPPING_CSV, dan keduanya
identik antara Authentic-Only dan Synthetic-Only (test set = 106 segmen
authentic yang sama persis, tidak berubah antar kondisi).

N=106 di sini bukan lagi "106 dari 5 fold rotasi" seperti Authentic-Only,
melainkan "106 dari 1 model yang sama, dites ke semua segmen sekaligus" --
tapi secara struktur CSV output-nya identik, jadi driver evaluasi ini juga
identik strukturnya.

Tambahan: bandingkan CI Synthetic-Only vs baseline Zero-Shot Whisper-small
(angka SAMA dgn yang dipakai evaluate_authentic_only_whisper.py, dari
rangkuman metodologi 9.8) -- operasionalisasi Klaim 1b untuk kondisi ini
(within-model, apakah fine-tuning ke data sintetis membantu, dibanding
Zero-Shot-nya sendiri -- BUKAN dibanding Authentic-Only, itu bukan klaim
utama tesis ini). CATATAN: overlap/non-overlap CI cuma indikasi kasar,
BUKAN uji hipotesis formal -- disebut eksplisit sebagai itu di tesis.

Prasyarat
---------
compute_wer_cer.py dan bootstrap_ci.py ada di folder yang sama (working
directory yang sama saat dijalankan di Colab), supaya import langsung jalan.

Gaya skrip: Colab (tanpa argparse), konsisten dgn skrip lain di pipeline ini.
"""

import json
from pathlib import Path

from compute_wer_cer import compute_wer, compute_cer
from bootstrap_ci import (
    bootstrap_ci,
    load_predictions_csv,
    load_recording_mapping,
    resolve_groups,
    print_grouping_diagnostic,
)

# ---- EDIT BAGIAN INI ----
PREDICTIONS_CSV = "/content/drive/MyDrive/bugis_authentic/synthetic_only_results/synthetic_only_whisper_predictions.csv"
RECORDING_MAPPING_CSV = "/content/drive/MyDrive/bugis_authentic/final_recording_mapping.csv"   # SAMA persis dgn Authentic-Only -- test set tidak berubah
OUTPUT_JSON = "/content/drive/MyDrive/bugis_authentic/synthetic_only_results/synthetic_only_whisper_bootstrap_ci.json"
GROUPING_DIAGNOSTIC_CSV = "/content/drive/MyDrive/bugis_authentic/synthetic_only_results/grouping_diagnostic_synthetic_only_whisper.csv"

N_BOOTSTRAP = 1000
CI_LEVEL = 0.95

# Baseline Zero-Shot Whisper-small -- ANGKA SAMA PERSIS dgn yang dipakai
# evaluate_authentic_only_whisper.py (rangkuman metodologi 9.8, N=106,
# bootstrap 95% CI group-aware, n_bootstrap=1000). Baseline pembanding
# selalu Zero-Shot untuk model yang sama, bukan Authentic-Only.
ZERO_SHOT_WHISPER_BASELINE = {
    "wer": {"point_estimate": 0.883, "ci_lower": 0.857, "ci_upper": 0.915},
    "cer": {"point_estimate": 0.228, "ci_lower": 0.219, "ci_upper": 0.237},
}
# --------------------------


def ci_overlap(a_lower: float, a_upper: float, b_lower: float, b_upper: float) -> bool:
    # bootstrap_ci() ngembaliin ci_lower/ci_upper sbg np.percentile (numpy.float64).
    # numpy.float64 kebetulan subclass Python float (json-safe), TAPI hasil
    # perbandingan numpy.float64 (<=) itu numpy.bool_ -- BUKAN subclass bool
    # Python, jadi json.dumps() bakal gagal kalau tidak di-cast eksplisit.
    return bool(a_lower <= b_upper and b_lower <= a_upper)


def main():
    file_ids, refs, hyps = load_predictions_csv(PREDICTIONS_CSV)
    n = len(file_ids)
    print(f"Synthetic-Only Whisper-small: {n} baris prediksi dimuat dari {PREDICTIONS_CSV}")
    if n != 106:
        print(f"  ⚠ Ekspektasi 106 baris (kalau training+eval selesai tanpa skip file), dapat {n} -- "
              f"cek log fine-tuning kalau ada file authentic yang di-skip karena error saat eval.")

    recording_mapping = load_recording_mapping(RECORDING_MAPPING_CSV)
    groups = resolve_groups(file_ids, recording_mapping)
    n_groups = len(set(groups))
    print(f"Rekaman unik terdeteksi: {n_groups} (ekspektasi 40, atau kurang kalau ada segmen yang skip)")

    Path(OUTPUT_JSON).parent.mkdir(parents=True, exist_ok=True)
    print_grouping_diagnostic(file_ids, groups, save_path=GROUPING_DIAGNOSTIC_CSV)

    results = {}
    print()
    for metric_name, fn in [("wer", compute_wer), ("cer", compute_cer)]:
        r = bootstrap_ci(
            refs, hyps, metric_fn=fn,
            n_bootstrap=N_BOOTSTRAP, ci=CI_LEVEL, groups=groups,
        )
        baseline = ZERO_SHOT_WHISPER_BASELINE[metric_name]
        overlap = ci_overlap(r["ci_lower"], r["ci_upper"], baseline["ci_lower"], baseline["ci_upper"])

        print(f"{metric_name.upper()}")
        print(f"  Synthetic-Only : {r['point_estimate']*100:.1f}% "
              f"[{r['ci_lower']*100:.1f}%, {r['ci_upper']*100:.1f}%] "
              f"(95% CI, grouped, n_groups={r['n_groups']})")
        print(f"  Zero-Shot      : {baseline['point_estimate']*100:.1f}% "
              f"[{baseline['ci_lower']*100:.1f}%, {baseline['ci_upper']*100:.1f}%]")
        print(f"  -> CI {'OVERLAP (belum tentu signifikan)' if overlap else 'TIDAK OVERLAP (indikasi signifikan)'}\n")

        r["zero_shot_baseline"] = baseline
        r["ci_overlap_with_zero_shot"] = overlap
        results[metric_name] = r

    Path(OUTPUT_JSON).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Tersimpan: {OUTPUT_JSON}")
    return results


if __name__ == "__main__":
    main()
