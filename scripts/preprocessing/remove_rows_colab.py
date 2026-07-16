# ============================================================
# Hapus baris tertentu dari sentences_for_tts.csv — sebelum generate TTS
# ============================================================

import csv
from pathlib import Path

# ---- EDIT BAGIAN INI ----
INPUT_CSV = "/content/sentences_for_tts.csv"
OUTPUT_CSV = "/content/sentences_for_tts_cleaned.csv"

# Isi (paragraph_id, sentence_index) yang mau dihapus — kombinasi keduanya
# harus persis match, jadi aman meski ada teks yang kebetulan mirip.
TO_REMOVE = {
    ("paragraph_029", "3"),   # "Amin"
    ("paragraph_062", "2"),   # "^ Holweck, Frederick." -- sitasi Wikipedia
    ("paragraph_200", "2"),   # "^ Flechner, Roy; Meeder, Sven, ed." -- sitasi Wikipedia
}
# --------------------------

kept_rows = []
removed_rows = []

with open(INPUT_CSV, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    for row in reader:
        key = (row["paragraph_id"], row["sentence_index"])
        if key in TO_REMOVE:
            removed_rows.append(row)
        else:
            kept_rows.append(row)

with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(kept_rows)

print(f"Baris asli: {len(kept_rows) + len(removed_rows)}")
print(f"Dihapus: {len(removed_rows)}")
for r in removed_rows:
    print(f"  - {r['file_stem']}: \"{r['text']}\"")
print(f"Sisa: {len(kept_rows)}")
print(f"\nDisimpan ke: {OUTPUT_CSV}")

found_keys = {(r["paragraph_id"], r["sentence_index"]) for r in removed_rows}
missing = TO_REMOVE - found_keys
if missing:
    print(f"\n⚠ Kombinasi berikut TIDAK ketemu di file, cek lagi ejaan/nomornya: {missing}")
else:
    print(f"\n✓ Semua {len(TO_REMOVE)} baris yang diminta berhasil ketemu dan dihapus.")
