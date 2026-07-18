# ============================================================
# Build manifest — versi Colab (tanpa argparse, tinggal edit path di bawah lalu Run)
# ============================================================

import csv
import re
from pathlib import Path

# ---- EDIT BAGIAN INI ----
AUDIO_DIR = "/content/drive/MyDrive/tts_output"   # folder isi file .mp3
TEXT_DIR = "/content/drive/MyDrive/tts_output"    # folder isi file .txt (boleh sama dengan AUDIO_DIR)
OUTPUT_CSV = "/content/synthetic_manifest.csv"
EXTENSIONS = (".mp3", ".wav")
# --------------------------


def natural_sort_key(filename: str):
    parts = re.split(r"(\d+)", filename)
    return [int(p) if p.isdigit() else p for p in parts]


def load_audio_files(audio_dir, extensions):
    audio_dir = Path(audio_dir)
    files = [f for f in audio_dir.iterdir() if f.suffix.lower() in extensions]
    files.sort(key=lambda f: natural_sort_key(f.name))
    return files


def build_manifest_from_pairs(audio_files, text_dir):
    text_dir = Path(text_dir)
    txt_files = {f.stem: f for f in text_dir.glob("*.txt")}

    rows = []
    unmatched_audio = []
    used_stems = set()

    for audio_file in audio_files:
        stem = audio_file.stem
        if stem in txt_files:
            text = txt_files[stem].read_text(encoding="utf-8-sig").strip()
            rows.append({"file_id": stem, "text": text, "audio_filename": audio_file.name})
            used_stems.add(stem)
        else:
            unmatched_audio.append(audio_file.name)

    orphan_txt = [f"{stem}.txt" for stem in txt_files if stem not in used_stems]
    return rows, unmatched_audio, orphan_txt


# ---- RUN ----
audio_files = load_audio_files(AUDIO_DIR, EXTENSIONS)
print(f"Found {len(audio_files)} audio files in {AUDIO_DIR}")

rows, unmatched_audio, orphan_txt = build_manifest_from_pairs(audio_files, TEXT_DIR)
print(f"Matched {len(rows)} audio<->text pairs by filename.")

if unmatched_audio:
    print(f"\n⚠ {len(unmatched_audio)} audio file(s) have NO matching .txt: {unmatched_audio[:10]}")
if orphan_txt:
    print(f"\n⚠ {len(orphan_txt)} .txt file(s) have NO matching audio: {orphan_txt[:10]}")
if not unmatched_audio and not orphan_txt:
    print("✓ Every audio file has exactly one matching .txt, and vice versa. Clean pairing.")

with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["file_id", "text", "audio_filename"])
    writer.writeheader()
    writer.writerows(rows)

print(f"\nWrote {len(rows)} pairs to {OUTPUT_CSV}")

print("\n=== First 3 pairs (VERIFY against actual audio) ===")
for row in rows[:3]:
    print(f"  {row['audio_filename']}  <->  {row['text'][:80]}")
print("\n=== Last 3 pairs (VERIFY these too) ===")
for row in rows[-3:]:
    print(f"  {row['audio_filename']}  <->  {row['text'][:80]}")
