# ============================================================
# Audit durasi audio — Authentic (GRN) vs Synthetic (TTS), versi Colab
# ============================================================
# pip install soundfile dulu kalau belum ada: !pip install soundfile -q

import soundfile as sf
from pathlib import Path

# ---- EDIT BAGIAN INI ----
AUTHENTIC_DIR = "/content/drive/MyDrive/grn_authentic"
SYNTHETIC_DIR = "/content/drive/MyDrive/tts_output"
EXTENSIONS = (".mp3", ".wav")
MIN_SEC = 1.0
MAX_SEC = 30.0   # batas keras Whisper
# --------------------------


def analyze_folder(label, folder, extensions, min_sec, max_sec):
    folder = Path(folder)
    files = [f for f in folder.iterdir() if f.suffix.lower() in extensions]

    durations = []
    too_short = []
    too_long = []
    failed = []

    for f in files:
        try:
            info = sf.info(str(f))
            duration = info.frames / info.samplerate
            durations.append((f.name, duration))
            if duration < min_sec:
                too_short.append((f.name, duration))
            if duration > max_sec:
                too_long.append((f.name, duration))
        except Exception as e:
            failed.append((f.name, str(e)))

    if failed:
        print(f"  ⚠ {len(failed)} file gagal dibaca: {failed[:5]}")

    if not durations:
        print(f"  Tidak ada file audio ditemukan di {folder}")
        return None

    vals = sorted(d for _, d in durations)
    n = len(vals)
    stats = {
        "label": label,
        "n": n,
        "min": vals[0],
        "max": vals[-1],
        "mean": sum(vals) / n,
        "median": vals[n // 2],
        "p90": vals[int(n * 0.9)],
        "too_short": too_short,
        "too_long": too_long,
    }

    print(f"\n=== {label} ({n} file) ===")
    print(f"  Min: {stats['min']:.1f}s | Max: {stats['max']:.1f}s | Mean: {stats['mean']:.1f}s | "
          f"Median: {stats['median']:.1f}s | P90: {stats['p90']:.1f}s")
    print(f"  Di bawah {min_sec}s: {len(too_short)} file")
    print(f"  Di atas {max_sec}s: {len(too_long)} file ({len(too_long)/n*100:.1f}%)")

    if too_long:
        with open(f"/content/duration_audit_{label.lower()}_too_long.csv", "w", encoding="utf-8") as out:
            out.write("filename,duration_sec\n")
            for name, dur in sorted(too_long, key=lambda x: -x[1]):
                out.write(f"{name},{dur:.2f}\n")
        print(f"  -> Daftar lengkap disimpan ke /content/duration_audit_{label.lower()}_too_long.csv")

    return stats


print("Menganalisis audio authentic dan synthetic...")
auth_stats = analyze_folder("Authentic", AUTHENTIC_DIR, EXTENSIONS, MIN_SEC, MAX_SEC)
synth_stats = analyze_folder("Synthetic", SYNTHETIC_DIR, EXTENSIONS, MIN_SEC, MAX_SEC)

if auth_stats and synth_stats:
    print(f"\n=== Perbandingan Authentic vs Synthetic ===")
    print(f"  {'Metrik':<12} {'Authentic':>12} {'Synthetic':>12}")
    print(f"  {'-'*12} {'-'*12} {'-'*12}")
    print(f"  {'Mean':<12} {auth_stats['mean']:>11.1f}s {synth_stats['mean']:>11.1f}s")
    print(f"  {'Median':<12} {auth_stats['median']:>11.1f}s {synth_stats['median']:>11.1f}s")
    print(f"  {'Min':<12} {auth_stats['min']:>11.1f}s {synth_stats['min']:>11.1f}s")
    print(f"  {'Max':<12} {auth_stats['max']:>11.1f}s {synth_stats['max']:>11.1f}s")

    ratio = synth_stats["mean"] / auth_stats["mean"] if auth_stats["mean"] else float("inf")
    print(f"\n  Rasio panjang rata-rata (synthetic/authentic): {ratio:.2f}x")
    if ratio > 1.5 or ratio < 0.67:
        print(f"  ⚠ Perbedaan cukup besar — worth dicatat di Limitasi sebagai confound "
              f"panjang-utterance antar kondisi, terpisah dari isu >30s di atas.")
