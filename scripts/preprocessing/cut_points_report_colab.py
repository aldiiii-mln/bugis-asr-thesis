# ============================================================
# Laporan titik potong (menit:detik) — untuk dipotong manual sendiri
# ============================================================
# pip install pydub dulu kalau belum ada: !pip install pydub -q

from pydub import AudioSegment
from pydub.silence import detect_silence
from pathlib import Path
import csv

# ---- EDIT BAGIAN INI ----
AUDIO_DIR = "/content/drive/MyDrive/grn_authentic"
OUTPUT_CSV = "/content/cut_points_report.csv"
MAX_SEGMENT_MS = 25000       # target maksimal per potongan (25s, margin dari batas 30s Whisper)
MIN_SILENCE_LEN_MS = 400
SILENCE_THRESH_DB = -40      # kalau titik potong kelihatan salah, coba naik/turunkan ini
EXTENSIONS = (".mp3", ".wav")
# --------------------------


def mmss(ms):
    total_sec = ms / 1000
    m = int(total_sec // 60)
    s = total_sec % 60
    return f"{m}:{s:05.2f}"


def find_cut_points(audio, min_silence_len, silence_thresh):
    silences = detect_silence(audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
    return sorted([(s + e) // 2 for s, e in silences])


def greedy_chunk(total_duration_ms, candidate_cuts, max_segment_ms):
    segments = []
    current_start = 0
    candidates = candidate_cuts + [total_duration_ms]
    i = 0
    while current_start < total_duration_ms:
        last_valid_cut = None
        while i < len(candidates) and candidates[i] - current_start <= max_segment_ms:
            last_valid_cut = candidates[i]
            i += 1
        if last_valid_cut is not None and last_valid_cut > current_start:
            segments.append((current_start, last_valid_cut, False))
            current_start = last_valid_cut
        else:
            forced_end = min(current_start + max_segment_ms, total_duration_ms)
            segments.append((current_start, forced_end, True))
            current_start = forced_end
            while i < len(candidates) and candidates[i] <= current_start:
                i += 1
    return segments


# ---- RUN ----
audio_dir = Path(AUDIO_DIR)
files = sorted([f for f in audio_dir.iterdir() if f.suffix.lower() in EXTENSIONS])
print(f"Menganalisis {len(files)} file (TIDAK memotong audio, cuma laporan titik potong)...\n")

report_rows = []

for f in files:
    audio = AudioSegment.from_file(f)
    duration_ms = len(audio)

    if duration_ms <= MAX_SEGMENT_MS:
        print(f"{f.name} ({mmss(duration_ms)}) — di bawah batas, tidak perlu dipotong\n")
        report_rows.append({
            "file": f.name, "segment": 1, "start": "0:00.00", "end": mmss(duration_ms),
            "duration_s": round(duration_ms/1000, 2), "forced_cut": False,
        })
        continue

    cut_points = find_cut_points(audio, MIN_SILENCE_LEN_MS, SILENCE_THRESH_DB)
    segments = greedy_chunk(duration_ms, cut_points, MAX_SEGMENT_MS)

    print(f"{f.name} (total {mmss(duration_ms)}):")
    for idx, (start, end, forced) in enumerate(segments, start=1):
        flag = "  ⚠ FORCED (bukan di jeda alami, dengarkan dulu sebelum potong di sini)" if forced else ""
        print(f"  Segmen {idx}: {mmss(start)} → {mmss(end)}  (durasi {(end-start)/1000:.1f}s){flag}")
        report_rows.append({
            "file": f.name, "segment": idx, "start": mmss(start), "end": mmss(end),
            "duration_s": round((end-start)/1000, 2), "forced_cut": forced,
        })
    print()

with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["file", "segment", "start", "end", "duration_s", "forced_cut"])
    writer.writeheader()
    writer.writerows(report_rows)

print(f"Laporan lengkap juga disimpan ke: {OUTPUT_CSV}")
