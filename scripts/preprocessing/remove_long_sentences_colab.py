# ============================================================
# Buang kalimat yang terlalu panjang (>30s) — audio + baris manifest
# ============================================================
# Aman untuk data yang SUDAH digenerate — file lain tidak disentuh/di-rename,
# nggak ada "urutan" yang bisa rusak karena penomoran boleh ada lubang.

import csv
from pathlib import Path

# ---- EDIT BAGIAN INI ----
AUDIO_DIR = "/content/drive/MyDrive/bugis_tts/audio"
MANIFEST_PATH = "/content/drive/MyDrive/bugis_tts/transkrip/synthetic_manifest.csv"
MANIFEST_OUTPUT = "/content/drive/MyDrive/bugis_tts/transkrip/synthetic_manifest_cleaned.csv"
VOICES = ["ardi", "gadis"]

# 62 index kalimat yang mau dibuang (dari analisis duration_audit_synthetic_too_long.csv)
INDICES_TO_REMOVE = [
    11, 13, 14, 15, 30, 33, 35, 36, 38, 41, 42, 52, 54, 64, 65, 72, 86, 93, 94,
    116, 122, 123, 130, 131, 146, 147, 149, 156, 172, 175, 176, 179, 181, 188,
    189, 190, 194, 195, 202, 213, 220, 221, 223, 224, 227, 229, 233, 240, 242,
    251, 258, 263, 264, 269, 294, 301, 302, 303, 304, 306, 307, 308,
]
# --------------------------


def build_file_ids(indices, voices, n_digits=3):
    ids = set()
    for idx in indices:
        for voice in voices:
            ids.add(f"{voice}{idx:0{n_digits}d}")
    return ids


# ---- HAPUS FILE AUDIO ----
audio_dir = Path(AUDIO_DIR)
file_ids_to_remove = build_file_ids(INDICES_TO_REMOVE, VOICES)

deleted = []
not_found = []
for file_id in sorted(file_ids_to_remove):
    matches = list(audio_dir.glob(f"{file_id}.*"))
    if matches:
        for m in matches:
            m.unlink()
            deleted.append(m.name)
    else:
        not_found.append(file_id)

print(f"Target dihapus: {len(file_ids_to_remove)} file_id ({len(INDICES_TO_REMOVE)} kalimat x {len(VOICES)} suara)")
print(f"Berhasil dihapus: {len(deleted)}")
if not_found:
    print(f"⚠ Tidak ketemu (mungkin sudah tidak ada / nama beda): {not_found}")

# ---- BERSIHKAN MANIFEST ----
kept_rows = []
removed_rows = []

with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    for row in reader:
        if row["file_id"] in file_ids_to_remove:
            removed_rows.append(row)
        else:
            kept_rows.append(row)

with open(MANIFEST_OUTPUT, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(kept_rows)

print(f"\nManifest: {len(removed_rows)} baris dihapus, {len(kept_rows)} baris tersisa")
print(f"Disimpan ke: {MANIFEST_OUTPUT}")

manifest_removed_ids = {r["file_id"] for r in removed_rows}
missing_in_manifest = file_ids_to_remove - manifest_removed_ids
if missing_in_manifest:
    print(f"\n⚠ file_id berikut ada di daftar hapus tapi TIDAK ketemu di manifest, cek manual: "
          f"{sorted(missing_in_manifest)}")
else:
    print(f"\n✓ Semua file_id yang dihapus dari audio juga konsisten terhapus dari manifest.")

print(f"\nPENTING: pakai {MANIFEST_OUTPUT} (bukan file manifest lama) untuk langkah "
      f"check_duplication.py dan training selanjutnya.")
