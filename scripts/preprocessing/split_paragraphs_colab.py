# ============================================================
# Pecah paragraf jadi kalimat individual — versi Colab (tanpa argparse)
# ============================================================
# Tujuan: dari 201 paragraf -> banyak kalimat pendek, siap di-generate ulang
# ke TTS supaya durasi audio per-file jauh di bawah batas 30 detik Whisper.

import csv
import re
from pathlib import Path


INPUT_PATH = "/content/sampel_201_final.csv"   # CSV (paragraph_id,text) ATAU .txt (1 paragraf per baris)
OUTPUT_PATH = "/content/sentences_for_tts.csv"



KNOWN_ABBREVIATIONS = {
    "dr", "mr", "mrs", "ms", "prof", "ir", "hj", "h", "drs", "st",
    "msc", "ma", "sh", "se", "kh", "pdt",
}


def split_into_sentences(paragraph: str) -> list[str]:
    """Pecah di titik/tanda seru/tanya. TIDAK mensyaratkan huruf besar
    setelahnya secara umum (korpus NusaTranslation kapitalisasinya tidak
    konsisten). Guard singkatan/gelar pakai DUA jenis sinyal presisi
    (bukan aturan umum "kata pendek = singkatan", yang ternyata salah
    tangkap kalimat pendek asli seperti "Ya."):
      1. Inisial satu huruf (mis. "R." pada "R. Leenawaty") — pola ini
         nyaris tidak ambigu.
      2. Kata dalam daftar singkatan gelar yang dikenal (Dr, Ir, Prof,
         Hj, dst.) — bukan sembarang kata pendek.
    Guard hanya aktif kalau kata SESUDAH titik juga berhuruf besar
    (pola "Gelar. Nama")."""
    pattern = re.compile(r'[.!?]+[”’"\']?')
    sentences = []
    start = 0
    for m in pattern.finditer(paragraph):
        end = m.end()
        before = paragraph[start:m.start()]
        after = paragraph[end:].lstrip()

        words_before = before.split()
        last_word = words_before[-1] if words_before else ""
        next_word = after.split()[0] if after.split() else ""

        is_single_initial = len(last_word) == 1 and last_word.isupper()
        is_known_abbrev = last_word.lower() in KNOWN_ABBREVIATIONS and last_word[:1].isupper()
        capitalized_after = next_word[:1].isupper() if next_word else False
        is_abbreviation = (is_single_initial or is_known_abbrev) and capitalized_after

        if is_abbreviation:
            continue  # jangan potong di sini, gabung ke titik berikutnya

        piece = paragraph[start:end].strip()
        if piece:
            sentences.append(piece)
        start = end

    trailing = paragraph[start:].strip()
    if trailing:
        sentences.append(trailing)
    return sentences


def load_paragraphs(path: str) -> list[tuple]:
    """Returns list of (original_id, text). Auto-detects CSV vs plain txt,
    and is flexible about the id column name (paragraph_id / index / id —
    whichever is present). Extra columns (e.g. kata_kunci_ditemukan) are
    simply ignored, not required."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            id_candidates = ["paragraph_id", "index", "id"]
            id_col = next((c for c in id_candidates if c in fieldnames), fieldnames[0])
            if "text" not in fieldnames:
                raise ValueError(f"No 'text' column found. Columns present: {fieldnames}")
            for row in reader:
                rows.append((row[id_col], row["text"]))
        return rows
    else:
        lines = path.read_text(encoding="utf-8").splitlines()
        return [(str(i + 1), line.strip()) for i, line in enumerate(lines) if line.strip()]


# ---- RUN ----
paragraphs = load_paragraphs(INPUT_PATH)
print(f"Loaded {len(paragraphs)} paragraphs from {INPUT_PATH}")

rows = []
word_counts = []
no_punctuation_paragraphs = []

for seq, (source_id, text) in enumerate(paragraphs, start=1):
    para_id = f"paragraph_{seq:03d}"   # clean sequential numbering for filenames
    sentences = split_into_sentences(text)

    if len(sentences) == 1 and not re.search(r'[.!?]', text):
        no_punctuation_paragraphs.append((para_id, source_id, text))

    for i, sent in enumerate(sentences, start=1):
        stem = f"{para_id}_sent{i:02d}"
        rows.append({
            "paragraph_id": para_id,
            "source_index": source_id,   # traceable back to original dataset row
            "sentence_index": i,
            "text": sent,
            "file_stem": stem,
        })
        word_counts.append(len(sent.split()))

with open(OUTPUT_PATH, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["paragraph_id", "source_index", "sentence_index", "text", "file_stem"])
    writer.writeheader()
    writer.writerows(rows)

print(f"\nMenghasilkan {len(rows)} kalimat dari {len(paragraphs)} paragraf "
      f"(rata-rata {len(rows)/len(paragraphs):.1f} kalimat/paragraf)")

if word_counts:
    wc_sorted = sorted(word_counts)
    n = len(wc_sorted)
    print(f"\n=== Statistik panjang kalimat (kata) ===")
    print(f"  Min: {wc_sorted[0]} | Max: {wc_sorted[-1]} | "
          f"Mean: {sum(wc_sorted)/n:.1f} | Median: {wc_sorted[n//2]}")

    very_short = [r for r in rows if len(r["text"].split()) <= 2]
    very_long = [r for r in rows if len(r["text"].split()) > 40]
    if very_short:
        print(f"\n⚠ {len(very_short)} kalimat sangat pendek (≤2 kata) — cek manual, "
              f"mungkin salah pecah (misal dialog terpotong):")
        for r in very_short[:5]:
            print(f"    {r['file_stem']}: \"{r['text']}\"")
    if very_long:
        print(f"\n⚠ {len(very_long)} kalimat masih panjang (>40 kata) — mungkin perlu "
              f"dicek manual, berpotensi tetap lama saat di-TTS:")
        for r in very_long[:5]:
            print(f"    {r['file_stem']}: \"{r['text'][:80]}...\"")

if no_punctuation_paragraphs:
    print(f"\n⚠⚠ {len(no_punctuation_paragraphs)} paragraf SAMA SEKALI TIDAK PUNYA tanda baca "
          f"(titik/seru/tanya) — tidak bisa dipecah otomatis dengan cara apa pun, karena "
          f"memang tidak ada tanda pemisah di teks sumbernya. Ini kemungkinan artefak dari "
          f"tahap scraping/filtering NusaTranslation sebelumnya, bukan bug di script ini:")
    for para_id, source_id, text in no_punctuation_paragraphs[:10]:
        print(f"    {para_id} (source_index={source_id}, {len(text.split())} kata): \"{text[:60]}...\"")
    no_punct_path = OUTPUT_PATH.replace(".csv", "_no_punctuation.csv")
    with open(no_punct_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["paragraph_id", "source_index", "text", "word_count"])
        for para_id, source_id, text in no_punctuation_paragraphs:
            writer.writerow([para_id, source_id, text, len(text.split())])
    print(f"    -> Daftar lengkap disimpan ke {no_punct_path}. Perlu diputuskan manual: "
          f"perbaiki tanda baca satu-satu, pecah paksa per-N-kata (berisiko motong di "
          f"tempat aneh), atau keluarkan dari pool synthetic.")

print(f"\nDisimpan ke {OUTPUT_PATH} — kolom 'file_stem' bisa langsung dipakai sebagai "
      f"nama file audio TTS (misal: paragraph_001_sent01_ardi.mp3), supaya tetap "
      f"tertelusur ke paragraf & kalimat asalnya.")
print(f"\nSelanjutnya: generate ulang TTS dari kolom 'text' di file ini (2 suara seperti "
      f"sebelumnya), lalu build_manifest.py mode 'pairs' seperti biasa.")
