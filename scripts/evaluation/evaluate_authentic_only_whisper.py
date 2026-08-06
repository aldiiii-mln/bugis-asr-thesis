"""
evaluate_authentic_only_whisper.py

Tujuan
------
Menutup loop item 5 (evaluasi) untuk Whisper-small di kondisi Authentic-Only,
dengan REUSE LANGSUNG compute_wer_cer.py + bootstrap_ci.py (import, BUKAN
disalin ulang) -- persis instruksi sesi ini.

Kenapa perlu driver terpisah (bukan langsung jalankan bootstrap_ci.py apa
adanya): __main__ bootstrap_ci.py hardcode pola nama file ala Zero-Shot
("zeroshot_{model_name}_predictions.csv" di dalam RESULTS_DIR, loop atas
MODEL_NAMES = ["whisper-small", "mms", "owsm-small"]). Output fine-tuning
Authentic-Only Whisper bernama authentic_only_whisper_predictions.csv --
tidak otomatis ketemu pola itu. Driver ini pakai fungsi-fungsi yang SAMA
persis (bootstrap_ci, load_predictions_csv, load_recording_mapping,
resolve_groups, print_grouping_diagnostic) lewat import, cuma titik masuknya
disesuaikan ke 1 file prediksi Authentic-Only.

Tambahan: langsung bandingkan CI Authentic-Only vs baseline Zero-Shot
Whisper-small (angka dari rangkuman metodologi 9.8) -- ini persis
operasionalisasi Klaim 1b (within-model, apakah fine-tuning benar-benar
memperbaiki model secara signifikan/CI tidak overlap, bukan kebetulan
sampel). CATATAN: overlap/non-overlap CI cuma indikasi kasar, BUKAN uji
hipotesis formal (mis. bukan paired test) -- baik dipakai sebagai sinyal
awal, disebut eksplisit sebagai itu di tesis, bukan p-value pengganti.

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
PREDICTIONS_CSV = "/content/drive/MyDrive/bugis_authentic/authentic_only_results/authentic_only_whisper_predictions.csv"
RECORDING_MAPPING_CSV = "/content/drive/MyDrive/bugis_authentic/final_recording_mapping.csv"
OUTPUT_JSON = "/content/drive/MyDrive/bugis_authentic/authentic_only_results/authentic_only_whisper_bootstrap_ci.json"
GROUPING_DIAGNOSTIC_CSV = "/content/drive/MyDrive/bugis_authentic/authentic_only_results/grouping_diagnostic_authentic_only_whisper.csv"

N_BOOTSTRAP = 1000
CI_LEVEL = 0.95

# Baseline Zero-Shot Whisper-small (dari rangkuman metodologi 9.8, N=106,
# bootstrap 95% CI group-aware, n_bootstrap=1000) -- untuk komparasi Klaim 1b.
# GANTI kalau baseline ini direvisi di sesi lain.
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
    print(f"Authentic-Only Whisper-small: {n} baris prediksi dimuat dari {PREDICTIONS_CSV}")
    if n != 106:
        print(f"  ⚠ Ekspektasi 106 baris (kalau training penuh 5 fold tanpa skip), dapat {n} -- "
              f"cek log fine-tuning kalau ada file yang di-skip karena error.")

    recording_mapping = load_recording_mapping(RECORDING_MAPPING_CSV)
    groups = resolve_groups(file_ids, recording_mapping)
    n_groups = len(set(groups))
    print(f"Rekaman unik terdeteksi: {n_groups} (ekspektasi 40, atau kurang dari 40 kalau ada fold yang skip semua segmen 1 rekaman)")

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
        print(f"  Authentic-Only : {r['point_estimate']*100:.1f}% "
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
