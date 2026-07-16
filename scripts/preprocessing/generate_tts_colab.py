# ============================================================
# Generate TTS sintetis — Edge TTS, urut per suara (Ardi dulu, baru Gadis)
# ============================================================
# pip install edge-tts nest_asyncio -q

import asyncio
import csv
from pathlib import Path

import edge_tts
import nest_asyncio

nest_asyncio.apply()  # perlu karena Colab/Jupyter sudah punya event loop sendiri

# ---- EDIT BAGIAN INI ----
# Kalau simpan ke Drive, mount dulu di cell terpisah:
#   from google.colab import drive
#   drive.mount('/content/drive')

INPUT_CSV = "/content/sentences_for_tts.csv"
AUDIO_DIR = "/content/drive/MyDrive/bugis_tts/audio"          # folder khusus audio
MANIFEST_DIR = "/content/drive/MyDrive/bugis_tts/transkrip"   # folder khusus manifest/teks

# Urutan list ini = urutan proses. Ardi diproses penuh dulu, baru Gadis.
# Rate bisa beda per suara kalau salah satu kedengaran terlalu cepat/lambat
# (format: "+0%" netral, "-10%" lebih lambat, "+10%" lebih cepat).
VOICES = [
    {"name": "ardi", "voice_id": "id-ID-ArdiNeural", "rate": "+0%"},
    {"name": "gadis", "voice_id": "id-ID-GadisNeural", "rate": "+0%"},
]
# --------------------------


def load_sentences(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


async def generate_one(text, voice_id, rate, output_path):
    communicate = edge_tts.Communicate(text, voice_id, rate=rate)
    await communicate.save(str(output_path))


async def generate_all(sentences, voices, output_dir, ext):
    failed = []
    skipped = []
    manifest_rows = []
    total = len(sentences) * len(voices)
    done = 0

    n_digits = max(3, len(str(len(sentences))))  # lebar angka menyesuaikan jumlah kalimat

    for voice in voices:
        print(f"\n=== Generate suara: {voice['name']} ({voice['voice_id']}, rate={voice['rate']}) ===")
        for idx, row in enumerate(sentences, start=1):
            text = row["text"]
            file_id = f"{voice['name']}{idx:0{n_digits}d}"
            output_path = output_dir / f"{file_id}.{ext}"
            done += 1

            if output_path.exists():
                skipped.append(output_path.name)
                print(f"  [{done}/{total}] (sudah ada, dilewati) {output_path.name}")
                manifest_rows.append({
                    "file_id": file_id, "voice": voice["name"], "text": text,
                    "source_paragraph_id": row.get("paragraph_id", ""),
                    "source_index": row.get("source_index", ""),
                })
                continue

            try:
                await generate_one(text, voice["voice_id"], voice["rate"], output_path)
                print(f"  [{done}/{total}] {output_path.name}")
                manifest_rows.append({
                    "file_id": file_id, "voice": voice["name"], "text": text,
                    "source_paragraph_id": row.get("paragraph_id", ""),
                    "source_index": row.get("source_index", ""),
                })
            except Exception as e:
                print(f"  [{done}/{total}] ⚠ GAGAL {output_path.name}: {e}")
                failed.append((file_id, voice["name"], text, str(e)))

    return failed, skipped, manifest_rows


# ---- RUN ----
audio_dir = Path(AUDIO_DIR)
manifest_dir = Path(MANIFEST_DIR)
audio_dir.mkdir(parents=True, exist_ok=True)
manifest_dir.mkdir(parents=True, exist_ok=True)
AUDIO_EXT = "mp3"

sentences = load_sentences(INPUT_CSV)
print(f"Loaded {len(sentences)} kalimat dari {INPUT_CSV}")
print(f"Target total file: {len(sentences) * len(VOICES)} ({len(sentences)} kalimat x {len(VOICES)} suara)")

failed, skipped, manifest_rows = asyncio.run(generate_all(sentences, VOICES, audio_dir, AUDIO_EXT))

manifest_path = manifest_dir / "synthetic_manifest.csv"
with open(manifest_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["file_id", "voice", "text", "source_paragraph_id", "source_index"])
    writer.writeheader()
    writer.writerows(manifest_rows)

print(f"\n=== Selesai ===")
print(f"Berhasil dibuat: {len(sentences)*len(VOICES) - len(failed) - len(skipped)}")
print(f"Dilewati (file sudah ada dari run sebelumnya): {len(skipped)}")
print(f"Gagal: {len(failed)}")
print(f"\nAudio disimpan di: {audio_dir}")
print(f"Manifest disimpan di: {manifest_path}")
print(f"(Nama file simpel: ardi001.mp3, gadis001.mp3, dst — ketertelusuran ke paragraf "
      f"asal tetap ada lewat kolom source_paragraph_id/source_index di manifest, "
      f"BUKAN build_manifest.py mode 'pairs' lagi — langsung pakai manifest ini "
      f"untuk check_duplication.py.)")

if failed:
    failed_csv = str(manifest_dir / "failed_generations.csv")
    with open(failed_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file_id", "voice", "text", "error"])
        writer.writerows(failed)
    print(f"\nDaftar yang gagal disimpan ke: {failed_csv}")
    print("Jalankan cell ini lagi untuk coba generate ulang yang gagal saja "
          "(yang sudah berhasil otomatis dilewati, tidak digenerate dobel).")
