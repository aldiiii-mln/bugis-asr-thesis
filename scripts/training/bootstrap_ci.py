"""
Bootstrap confidence intervals for WER/CER (Section 4.3 / 5). Applied
consistently to both metrics — no double standard between them.

Especially important for small test sets (Authentic-Only, N=5-ish per fold)
where a point-estimate WER can be misleadingly precise-looking.

UPDATE: sekarang GROUP-AWARE per rekaman asli (bukan per segmen). 106 segmen
authentic itu cuma berasal dari 40 rekaman independen (rata-rata ~2.65
segmen/rekaman) — segmen dari rekaman yang sama kemungkinan berkorelasi
(kondisi akustik & gaya bicara sama), persis catatan k-fold group-aware di
metodologi bagian 4.2. Kalau resampling per-segmen biasa (versi lama),
CI-nya kelihatan lebih sempit/pasti dari yang sebenarnya karena treat 106
"sampel independen" padahal aslinya cuma 40 titik data independen.

Cluster/block bootstrap: yang di-resample adalah REKAMAN (with replacement),
lalu semua segmen dari rekaman yang terpilih ikut bareng dalam satu resample
— bukan segmen dipilih satu-satu.
"""

import re
import csv
import json
from pathlib import Path

import numpy as np
from compute_wer_cer import compute_wer, compute_cer
# compute_wer_cer.py sudah diupload & diperbaiki (fix truth_transform->reference_transform
# utk jiwer>=4.0, dan fix CER supaya case-insensitive sama seperti WER) — sudah dites
# end-to-end bareng file ini, hasilnya konsisten.



def extract_recording_id(file_id: str) -> str:
    """Ambil ID rekaman asli dari nama segmen, mis. 'grn01_seg01' -> 'grn01'.
    CATATAN: untuk data authentic saat ini, format file_id TERNYATA cuma
    penomoran global (1-107), bukan 'grnNN_segNN' — fungsi ini gagal untuk
    kasus itu (semua jadi grup sendiri). Dipertahankan sebagai fallback
    generik untuk manifest lain yang mungkin memang pakai format prefix.
    UNTUK DATA AUTHENTIC SEKARANG, PAKAI load_recording_mapping() DI BAWAH."""
    match = re.match(r"^(.+)_seg\d+$", file_id)
    if match:
        return match.group(1)
    return file_id  # fallback: kalau pola tidak cocok, jadi grup sendiri


def load_recording_mapping(path):
    """Baca file_id -> recording_id dari final_recording_mapping.csv.
    Mapping ini direkonstruksi dari penanda teks "Gambara N" di transkrip
    (bukan dari nama file — file_id di manifest ternyata cuma penomoran
    global 1-107, bukan prefix nama rekaman), dan sudah diverifikasi silang
    lewat pengecekan audio manual + kecocokan jumlah segmen per rekaman ke
    cut_points_report.csv (39/40 cocok persis, 1 beda krn segmen 3+4 di
    rekaman 12 digabung manual jadi 1 segmen — sudah dikonfirmasi benar)."""
    mapping = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mapping[row["file_id"]] = row["recording_id"]
    return mapping


def resolve_groups(file_ids, recording_mapping=None):
    """Tentukan grup (rekaman) per file_id. Prioritas: lookup dari
    recording_mapping (sudah diverifikasi) kalau tersedia; fallback ke
    extract_recording_id() regex kalau tidak."""
    if recording_mapping is None:
        return [extract_recording_id(fid) for fid in file_ids]

    groups = []
    missing = []
    for fid in file_ids:
        if fid in recording_mapping:
            groups.append(recording_mapping[fid])
        else:
            missing.append(fid)
            groups.append(fid)  # fallback: jadi grup sendiri kalau nggak ketemu di mapping
    if missing:
        preview = missing[:10]
        suffix = "..." if len(missing) > 10 else ""
        print(f"  ⚠ {len(missing)} file_id TIDAK ADA di recording_mapping (dianggap grup sendiri): {preview}{suffix}")
    return groups


def bootstrap_ci(
    references: list[str],
    hypotheses: list[str],
    metric_fn=compute_wer,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
    groups: list[str] | None = None,
) -> dict:
    """
    groups: list ID rekaman per segmen (sepanjang references/hypotheses).
      - Kalau diisi -> cluster/block bootstrap, resample REKAMAN (grouped).
        WAJIB dipakai untuk data authentic (106 segmen dari 40 rekaman).
      - Kalau None -> resampling per-item biasa (perilaku lama). Hanya valid
        kalau item-itemnya memang independen satu sama lain (mis. potensial
        untuk Synthetic-Only per kalimat unik — TAPI cek juga groupingnya
        kalau ada pasangan 2 suara per kalimat, itu juga bentuk korelasi).
    """
    assert len(references) == len(hypotheses)
    n = len(references)
    rng = np.random.default_rng(seed)

    point_estimate = metric_fn(references, hypotheses)

    if groups is None:
        scores = []
        for _ in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            r_sample = [references[i] for i in idx]
            h_sample = [hypotheses[i] for i in idx]
            scores.append(metric_fn(r_sample, h_sample))
        n_groups = n
    else:
        assert len(groups) == n
        unique_groups = sorted(set(groups))
        n_groups = len(unique_groups)
        group_to_indices: dict[str, list[int]] = {}
        for i, g in enumerate(groups):
            group_to_indices.setdefault(g, []).append(i)

        scores = []
        for _ in range(n_bootstrap):
            sampled_groups = rng.choice(unique_groups, size=n_groups, replace=True)
            idx = []
            for g in sampled_groups:
                idx.extend(group_to_indices[g])
            r_sample = [references[i] for i in idx]
            h_sample = [hypotheses[i] for i in idx]
            scores.append(metric_fn(r_sample, h_sample))

    alpha = (1 - ci) / 2
    lower = np.percentile(scores, alpha * 100)
    upper = np.percentile(scores, (1 - alpha) * 100)

    return {
        "point_estimate": point_estimate,
        "ci_lower": lower,
        "ci_upper": upper,
        "ci_level": ci,
        "n_bootstrap": n_bootstrap,
        "n_samples": n,
        "n_groups": n_groups,
        "grouped": groups is not None,
    }


def load_predictions_csv(path):
    """Baca CSV hasil evaluate_model() di zero_shot_eval_*.py
    (kolom: file_id, reference, hypothesis)."""
    file_ids, refs, hyps = [], [], []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            file_ids.append(row["file_id"])
            refs.append(row["reference"])
            hyps.append(row["hypothesis"])
    return file_ids, refs, hyps


def print_grouping_diagnostic(file_ids, groups, save_path=None):
    """Tampilkan ringkasan rekaman->segmen, dan opsional simpan rincian
    lengkapnya ke CSV — supaya bisa dicek manual bahwa extract_recording_id()
    memecah file_id dengan benar (bukan cuma percaya angka n_groups)."""
    from collections import defaultdict
    rec_to_segments = defaultdict(list)
    for fid, g in zip(file_ids, groups):
        rec_to_segments[g].append(fid)

    counts = sorted(len(v) for v in rec_to_segments.values())
    print(f"  Rincian grouping: {len(rec_to_segments)} rekaman, "
          f"segmen/rekaman min={counts[0]}, max={counts[-1]}, rata-rata={sum(counts)/len(counts):.2f}")

    if save_path:
        with open(save_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["recording_id", "n_segments", "segment_file_ids"])
            for rec_id in sorted(rec_to_segments):
                segs = rec_to_segments[rec_id]
                writer.writerow([rec_id, len(segs), ";".join(segs)])
        print(f"  Rincian lengkap per rekaman disimpan ke: {save_path} (buka & cek manual sebelum percaya n_groups di atas)")


# ---- RUN (gaya Colab, tanpa argparse — lihat metodologi 9.7) ----
if __name__ == "__main__":
    # ---- EDIT BAGIAN INI ----
    RESULTS_DIR = "/content/zero_shot_results"   # folder CSV hasil zero_shot_eval_*.py
    MODEL_NAMES = ["whisper-small", "mms", "owsm-small"]  # cocok dg nama file zeroshot_{name}_predictions.csv
    RECORDING_MAPPING_CSV = "/content/final_recording_mapping.csv"  # sudah diverifikasi (teks + manual, lihat metodologi)
                                                                      # set None kalau mau pakai extract_recording_id() regex biasa
    N_BOOTSTRAP = 1000
    CI_LEVEL = 0.95
    SAVE_GROUPING_DIAGNOSTIC = True   # simpan CSV rincian rekaman->segmen per model, buat double-check manual
    # --------------------------

    recording_mapping = load_recording_mapping(RECORDING_MAPPING_CSV) if RECORDING_MAPPING_CSV else None
    if recording_mapping is not None:
        print(f"Pakai recording_mapping dari {RECORDING_MAPPING_CSV} ({len(recording_mapping)} entri)\n")
    else:
        print("⚠ RECORDING_MAPPING_CSV=None — pakai extract_recording_id() regex (kemungkinan tidak akurat untuk data ini)\n")

    all_results = {}
    for model_name in MODEL_NAMES:
        csv_path = Path(RESULTS_DIR) / f"zeroshot_{model_name}_predictions.csv"
        if not csv_path.exists():
            print(f"⚠ Skip {model_name}: {csv_path} tidak ditemukan")
            continue

        file_ids, refs, hyps = load_predictions_csv(csv_path)
        groups = resolve_groups(file_ids, recording_mapping)
        n_groups = len(set(groups))
        print(f"=== {model_name}: N={len(file_ids)} segmen, {n_groups} rekaman unik terdeteksi ===")
        if n_groups > 45 or n_groups < 35:
            print(f"  ⚠ Jumlah rekaman terdeteksi ({n_groups}) jauh dari ekspektasi ~40 — "
                  f"cek RECORDING_MAPPING_CSV atau pola file_id.")

        diag_path = Path(RESULTS_DIR) / f"grouping_diagnostic_{model_name}.csv" if SAVE_GROUPING_DIAGNOSTIC else None
        print_grouping_diagnostic(file_ids, groups, save_path=diag_path)

        model_result = {}
        for metric_name, fn in [("wer", compute_wer), ("cer", compute_cer)]:
            result = bootstrap_ci(
                refs, hyps, metric_fn=fn,
                n_bootstrap=N_BOOTSTRAP, ci=CI_LEVEL, groups=groups,
            )
            model_result[metric_name] = result
            print(f"  {metric_name.upper()}: {result['point_estimate']*100:.1f}% "
                  f"[{result['ci_lower']*100:.1f}%, {result['ci_upper']*100:.1f}%] "
                  f"(95% CI, grouped, n_groups={result['n_groups']})")
        all_results[model_name] = model_result
        print()

    out_path = Path(RESULTS_DIR) / "bootstrap_ci_summary.json"
    out_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"Ringkasan lengkap disimpan ke: {out_path}")