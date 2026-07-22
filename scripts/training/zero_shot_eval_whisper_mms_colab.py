# ============================================================
# Zero-Shot Evaluation — Whisper-small & MMS, TANPA fine-tuning
# ============================================================
# pip install transformers torch librosa jiwer soundfile accelerate -q
#
# CATATAN: bagian pemanggilan model (transcribe_whisper, transcribe_mms)
# ditulis mengikuti API standar HuggingFace transformers, TAPI belum
# bisa saya tes end-to-end di sandbox saya (tidak ada akses ke model hub
# atau file audio asli). Struktur kode & alur logikanya solid, tapi
# jalankan dulu ke 2-3 file sebagai smoke test sebelum full run ke 106.
#
# File ini SENGAJA dipisah dari OWSM (lihat zero_shot_eval_owsm_colab.py)
# karena stack dependency beda (transformers vs espnet) — kalau salah satu
# bermasalah pas instalasi/run, yang lain nggak ikut kena.

import csv
import unicodedata
from pathlib import Path

import torch
import librosa
import jiwer

# ---- EDIT BAGIAN INI ----
MANIFEST_PATH = "/content/authentic_manifest.csv"       # hasil build_manifest.py, kolom: file_id, text, audio_filename
AUDIO_DIR = "/content/drive/MyDrive/bugis_authentic/audio"
OUTPUT_DIR = "/content/zero_shot_results"                # SAMA dengan OUTPUT_DIR di script OWSM, biar hasil ketiganya kekumpul jadi satu folder

WHISPER_MODEL_SIZE = "small"     # ganti "medium" untuk model kedua, kalau nanti diputuskan pakai Whisper-medium (bukan OWSM)
MMS_CHECKPOINT = "facebook/mms-1b-all"
MMS_TARGET_LANG = "ind"          # adapter Indonesian, dikonfirmasi tersedia (lihat metodologi 4.1)

SAMPLE_RATE = 16000
SMOKE_TEST_N = None                 # set None untuk full run ke semua data
# --------------------------


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text.strip())
    return " ".join(text.split())


def load_manifest(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


# ---------------- WHISPER ----------------

def load_whisper(model_size):
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    model_name = f"openai/whisper-{model_size}"
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()
    return model, processor


def transcribe_whisper(model, processor, audio_path, language="id"):
    audio, _ = librosa.load(str(audio_path), sr=SAMPLE_RATE)
    inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    input_features = inputs.input_features
    if torch.cuda.is_available():
        input_features = input_features.to("cuda")
    forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task="transcribe")
    with torch.no_grad():
        predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
    return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]


# ---------------- MMS ----------------

def load_mms(target_lang):
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    processor = Wav2Vec2Processor.from_pretrained(MMS_CHECKPOINT, target_lang=target_lang)
    model = Wav2Vec2ForCTC.from_pretrained(MMS_CHECKPOINT, target_lang=target_lang, ignore_mismatched_sizes=True)
    model.load_adapter(target_lang)
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()
    return model, processor


def transcribe_mms(model, processor, audio_path):
    audio, _ = librosa.load(str(audio_path), sr=SAMPLE_RATE)
    inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    input_values = inputs.input_values
    if torch.cuda.is_available():
        input_values = input_values.to("cuda")
    with torch.no_grad():
        logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    return processor.batch_decode(predicted_ids)[0]


# ---------------- EVALUASI ----------------

def evaluate_model(model_name, transcribe_fn, manifest_rows, audio_dir, output_dir):
    results = []
    audio_dir = Path(audio_dir)

    for row in manifest_rows:
        audio_path = audio_dir / row["audio_filename"]
        reference = normalize(row["text"])
        try:
            hypothesis = normalize(transcribe_fn(audio_path))
        except Exception as e:
            print(f"  ⚠ GAGAL {row['file_id']}: {e}")
            continue
        results.append({"file_id": row["file_id"], "reference": reference, "hypothesis": hypothesis})
        print(f"  {row['file_id']}")
        print(f"    ref: {reference[:70]}")
        print(f"    hyp: {hypothesis[:70]}")

    if not results:
        print(f"  ⚠ Tidak ada hasil untuk {model_name} — cek error di atas.")
        return None

    refs = [r["reference"] for r in results]
    hyps = [r["hypothesis"] for r in results]
    wer = jiwer.wer(refs, hyps)
    cer = jiwer.cer(refs, hyps)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"zeroshot_{model_name}_predictions.csv"
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file_id", "reference", "hypothesis"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\n=== {model_name}: WER={wer:.4f} ({wer*100:.1f}%), CER={cer:.4f} ({cer*100:.1f}%), N={len(results)} ===")
    print(f"Prediksi lengkap disimpan ke: {output_path}\n")
    return {"model": model_name, "wer": wer, "cer": cer, "n": len(results)}


# ---- RUN ----
manifest_rows = load_manifest(MANIFEST_PATH)
if SMOKE_TEST_N:
    print(f"⚠ MODE SMOKE TEST — cuma {SMOKE_TEST_N} file dulu. Set SMOKE_TEST_N = None untuk full run.\n")
    manifest_rows = manifest_rows[:SMOKE_TEST_N]

print(f"Total data evaluasi: {len(manifest_rows)} segmen\n")

summary = []

print(f"=== Whisper-{WHISPER_MODEL_SIZE} (zero-shot) ===")
whisper_model, whisper_processor = load_whisper(WHISPER_MODEL_SIZE)
result = evaluate_model(
    f"whisper-{WHISPER_MODEL_SIZE}",
    lambda path: transcribe_whisper(whisper_model, whisper_processor, path),
    manifest_rows, AUDIO_DIR, OUTPUT_DIR,
)
if result:
    summary.append(result)
del whisper_model  # bebaskan memori GPU sebelum load model berikutnya
torch.cuda.empty_cache() if torch.cuda.is_available() else None

print(f"=== MMS (zero-shot, adapter '{MMS_TARGET_LANG}') ===")
mms_model, mms_processor = load_mms(MMS_TARGET_LANG)
result = evaluate_model(
    "mms",
    lambda path: transcribe_mms(mms_model, mms_processor, path),
    manifest_rows, AUDIO_DIR, OUTPUT_DIR,
)
if result:
    summary.append(result)
del mms_model
torch.cuda.empty_cache() if torch.cuda.is_available() else None

print("\n=== RINGKASAN ZERO-SHOT (Whisper + MMS) ===")
for s in summary:
    print(f"  {s['model']}: WER={s['wer']*100:.1f}%, CER={s['cer']*100:.1f}%, N={s['n']}")
print("\nCatatan: jalankan zero_shot_eval_owsm_colab.py secara terpisah untuk model ketiga (OWSM-small).")