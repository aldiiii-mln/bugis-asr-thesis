"""
finetune_whisper_synthetic_colab.py

Tujuan
------
Fine-tuning Whisper-small ke kondisi Synthetic-Only: training memakai SEMUA
504 file sintetis sekaligus (TANPA k-fold), lalu evaluasi LANGSUNG ke SEMUA
106 segmen authentic (test set yang SAMA dengan Zero-Shot & Authentic-Only).

============================================================================
PERBEDAAN STRUKTURAL vs finetune_whisper_authentic_colab.py (baca dulu)
============================================================================
- TIDAK ADA loop fold -- training (sintetis) dan test (authentic) sumbernya
  beda total, jadi tidak ada risiko kebocoran seperti Authentic-Only (yang
  data authentic-nya dipakai dobel sbg train DAN test lewat rotasi fold).
- Konsekuensi: cuma 1 model dilatih (bukan 5 model per fold yang dibuang) --
  model ini DISIMPAN ke Drive di akhir run (beda dari Authentic-Only yang
  modelnya sengaja dibuang tiap fold, cuma prediksi mentah yang disimpan).
- Resume-capable, TAPI mekanismenya beda dari Authentic-Only: karena cuma
  ada 1 training run panjang (bukan 5 fold pendek terpisah), resume training
  memakai checkpoint HF Trainer (save_strategy="steps", tinggal
  resume_from_checkpoint=True kalau checkpoint lama ada di OUTPUT_DIR) --
  bukan skip per-fold seperti Authentic-Only. Skrip ini otomatis lanjut dari
  checkpoint terakhir kalau OUTPUT_DIR/checkpoint-* sudah ada, dan skip
  training+eval sama sekali kalau model final (FINAL_MODEL_DIR) + prediksi
  sudah tersimpan.
- Evaluasi TETAP wajib group-aware bootstrap CI per rekaman (dihitung
  terpisah oleh evaluate_synthetic_only_whisper.py, bukan bagian skrip ini)
  -- 106 segmen authentic tetap cuma 40 rekaman independen.

============================================================================
KONSISTENSI HYPERPARAMETER (WAJIB SAMA PERSIS dgn Authentic-Only Whisper)
============================================================================
MAX_STEPS=600, learning_rate=1e-5, warmup_steps=60, per_device_batch=8,
grad_accum=2 (effective 16), fp16=True, gradient_checkpointing=True --
disalin PERSIS dari finetune_whisper_authentic_colab.py (variabel kontrol,
lihat ringkasan_konfigurasi_fine_tuning.md & rangkuman_metodologi 10.2).
JANGAN diubah sepihak di sini -- kalau MAX_STEPS Whisper direvisi, revisi di
KEDUA skrip (Authentic-Only & Synthetic-Only), bukan cuma salah satu.

Implikasi epoch efektif (EXPECTED, bagian dari apa yang mau diuji tesis):
600 step x 16 (effective batch) / 504 (file sintetis) ~ 19 epoch efektif --
jauh lebih rendah dari Authentic-Only (~113 epoch efektif ke 85 segmen
train/fold), karena compute budget sama tapi data jauh lebih banyak. Skrip
ini mencetak angka ini di runtime (bukan cuma di komentar) supaya kelihatan
langsung tiap dijalankan.

Decoding evaluasi: greedy, language="id", task="transcribe", TANPA
max_length/num_beams eksplisit -- DICOCOKKAN PERSIS ke
zero_shot_eval_whisper_mms_colab.py & finetune_whisper_authentic_colab.py,
supaya gap hasil murni efek fine-tuning, bukan confound strategi decoding.

============================================================================
CATATAN -- KOLOM AUDIO DI synthetic_manifest_cleaned.csv
============================================================================
Manifest sintetis HANYA punya kolom: file_id, voice, text, source_paragraph_id,
source_index -- TIDAK ADA kolom audio_filename/path eksplisit (beda dari
authentic_manifest.csv yang punya kolom audio_filename langsung). Skrip ini
membentuk nama file audio = file_id + SYNTHETIC_AUDIO_EXT (mis. "ardi001" +
".mp3" = "ardi001.mp3") lewat fungsi synthetic_audio_path() di bawah.
DIKONFIRMASI user lewat screenshot Google Drive (folder bugis_tts/audio):
  - Ekstensi = .mp3 (bukan .wav)
  - Struktur = 1 folder FLAT, ardi001.mp3...ardi314.mp3 dan gadis001.mp3...
    tergabung di folder yang sama (bukan subfolder terpisah per suara) --
    cocok dgn asumsi awal, SYNTHETIC_AUDIO_DIR sudah diisi path yang benar.
JALANKAN SMOKE_TEST_N (mis. 3) DULU sebelum full run tetap disarankan --
sanity check umum sebelum commit ke run 600 step (bisa 1+ jam), bukan lagi
karena keraguan soal path/ekstensi audio.

PERINGATAN RESUME vs SMOKE TEST: JANGAN pakai OUTPUT_DIR yang sama untuk
smoke test dan full run -- kalau smoke test sempat menyimpan FINAL_MODEL_DIR
atau predictions CSV duluan, full run berikutnya akan mengira semuanya
sudah selesai dan skip total. Pakai OUTPUT_DIR terpisah untuk smoke test,
atau hapus isi OUTPUT_DIR sebelum full run.

Prasyarat sebelum run di Colab
-------------------------------
- Google Drive sudah di-mount.
- synthetic_manifest_cleaned.csv, authentic_manifest.csv sudah diupload ke
  folder Drive yang sesuai CONFIG di bawah (SYNTHETIC_MANIFEST_PATH masih
  perlu dicek -- lokasi persisnya belum dikonfirmasi user, cuma folder audio
  bugis_tts/audio yang sudah dikonfirmasi).
- 504 file audio sintetis ada di bugis_tts/audio (dikonfirmasi), 106 file
  audio authentic sudah diupload/di-mount di AUTHENTIC_AUDIO_DIR.
- pip install: transformers, datasets, accelerate, librosa, soundfile

Gaya skrip: Colab (tanpa argparse), konsisten dgn skrip lain di pipeline ini.
"""

import os
import gc
import json
import inspect
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import librosa

from dataclasses import dataclass
from typing import Any, Dict, List, Union

from datasets import Dataset
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    set_seed,
)

# ---- EDIT BAGIAN INI ----
SYNTHETIC_MANIFEST_PATH = "/content/drive/MyDrive/bugis_authentic/synthetic_manifest_cleaned.csv"
SYNTHETIC_AUDIO_DIR = "/content/drive/MyDrive/bugis_tts/audio"   # DIKONFIRMASI user (screenshot Drive): folder flat, ardi+gadis gabung
SYNTHETIC_AUDIO_EXT = ".mp3"    # DIKONFIRMASI user -- ardi001.mp3 dst

AUTHENTIC_MANIFEST_PATH = "/content/drive/MyDrive/bugis_authentic/authentic_manifest.csv"   # test set TIDAK berubah dari Zero-Shot/Authentic-Only
AUTHENTIC_AUDIO_DIR = "/content/drive/MyDrive/bugis_authentic/audio"    # SAMA dgn zero_shot_eval_whisper_mms_colab.py

OUTPUT_DIR = "/content/drive/MyDrive/bugis_authentic/synthetic_only_results"
FINAL_MODEL_DIR = os.path.join(OUTPUT_DIR, "final_model")   # model DISIMPAN di sini (beda dari Authentic-Only)

MODEL_NAME = "openai/whisper-small"
LANGUAGE = "id"              # SAMA PERSIS dgn zero-shot & Authentic-Only (bukan "indonesian")
TASK = "transcribe"

RANDOM_SEED = 42             # single-seed, konsisten dgn Zero-Shot & Authentic-Only

# ---- Compute budget: SAMA PERSIS dgn finetune_whisper_authentic_colab.py (variabel kontrol) ----
MAX_STEPS = 600
LEARNING_RATE = 1e-5
WARMUP_STEPS = 60
PER_DEVICE_TRAIN_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 2     # effective batch = 16
LOGGING_STEPS = 20

# ---- Checkpoint periodik (BARU dibanding Authentic-Only -- di sana save_strategy="no"
#      krn cuma butuh prediksi akhir per fold; di sini training 1x panjang & modelnya
#      mau disimpan, jadi butuh titik resume mid-training juga) ----
SAVE_STEPS = 100
SAVE_TOTAL_LIMIT = 2         # buang checkpoint lama otomatis, hemat kuota Drive

FP16 = True
GRADIENT_CHECKPOINTING = True

SMOKE_TEST_N = None          # set angka kecil (mis. 3) utk smoke test dulu -- lihat PERINGATAN di atas
# --------------------------


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """SAMA PERSIS dgn finetune_whisper_authentic_colab.py -- padding
    input_features (mel spectrogram) & labels (token ids) terpisah, label
    padding di-mask jadi -100 supaya tidak ikut dihitung loss."""
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def normalize(text: str) -> str:
    """SAMA PERSIS dgn zero_shot_eval_whisper_mms_colab.py / finetune_whisper_authentic_colab.py."""
    text = unicodedata.normalize("NFC", text.strip())
    return " ".join(text.split())


def load_audio(path: str) -> np.ndarray:
    array, _sr = librosa.load(path, sr=16000, mono=True)
    return array


def synthetic_audio_path(file_id: str) -> str:
    """Bentuk path audio sintetis dari file_id -- lihat catatan asumsi
    SYNTHETIC_AUDIO_EXT/SYNTHETIC_AUDIO_DIR di docstring atas. Diisolasi jadi
    1 fungsi supaya gampang diedit kalau ternyata strukturnya beda (mis.
    subfolder terpisah per suara ardi/ vs gadis/)."""
    return os.path.join(SYNTHETIC_AUDIO_DIR, file_id + SYNTHETIC_AUDIO_EXT)


def build_synthetic_dataset(df: pd.DataFrame, processor: WhisperProcessor) -> Dataset:
    """Konversi manifest sintetis (file_id, text) jadi HF Dataset berisi
    input_features (mel spectrogram) + labels (token ids)."""
    ds = Dataset.from_pandas(df[["file_id", "text"]].reset_index(drop=True))

    def _prepare(example):
        audio_array = load_audio(synthetic_audio_path(example["file_id"]))
        example["input_features"] = processor.feature_extractor(
            audio_array, sampling_rate=16000
        ).input_features[0]
        example["labels"] = processor.tokenizer(example["text"]).input_ids
        return example

    ds = ds.map(_prepare, remove_columns=["text"])
    return ds


def transcribe_whisper_one(model, processor, forced_decoder_ids, audio_path, device) -> str:
    """SAMA PERSIS dgn finetune_whisper_authentic_colab.py / zero-shot."""
    audio, _ = librosa.load(str(audio_path), sr=16000)
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features
    if device == "cuda":
        input_features = input_features.to("cuda")
    with torch.no_grad():
        predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
    return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]


def predict_all(model, processor, forced_decoder_ids, test_df: pd.DataFrame, device) -> List[Dict]:
    """Generate prediksi mentah utk SEMUA segmen authentic, PER-FILE dengan
    try/except -- meniru pola evaluate_model()/predict_fold(), 1 file gagal
    tidak menggagalkan seluruh evaluasi."""
    model.eval()
    results = []
    rows = test_df.reset_index(drop=True)

    for row in rows.itertuples():
        audio_path = os.path.join(AUTHENTIC_AUDIO_DIR, row.audio_filename)
        reference = normalize(row.text)
        try:
            hypothesis = normalize(
                transcribe_whisper_one(model, processor, forced_decoder_ids, audio_path, device)
            )
        except Exception as e:
            print(f"  \u26a0 GAGAL file_id={row.file_id}: {e}")
            continue

        results.append({
            "file_id": row.file_id,
            "reference": reference,
            "hypothesis": hypothesis,
        })
        print(f"  file_id={row.file_id}: ref='{reference[:60]}' | hyp='{hypothesis[:60]}'")

    return results


def main():
    set_seed(RANDOM_SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("PERINGATAN: GPU tidak terdeteksi -- training di CPU akan SANGAT lambat. "
              "Pastikan runtime Colab diset ke GPU (Runtime > Change runtime type > T4 GPU).")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    synth_df = pd.read_csv(SYNTHETIC_MANIFEST_PATH)
    assert len(synth_df) == 504, f"Expected 504 baris sintetis, dapat {len(synth_df)} -- cek SYNTHETIC_MANIFEST_PATH"

    auth_df = pd.read_csv(AUTHENTIC_MANIFEST_PATH)
    assert len(auth_df) == 106, f"Expected 106 baris authentic, dapat {len(auth_df)} -- cek AUTHENTIC_MANIFEST_PATH"

    if SMOKE_TEST_N:
        print(f"\u26a0 MODE SMOKE TEST -- training cuma {SMOKE_TEST_N} file sintetis, "
              f"eval cuma {SMOKE_TEST_N} file authentic. Set SMOKE_TEST_N = None utk full run.\n"
              f"\u26a0 INGAT: pakai OUTPUT_DIR terpisah dari full run (lihat PERINGATAN di docstring atas).\n")
        synth_df = synth_df.iloc[:SMOKE_TEST_N].reset_index(drop=True)
        auth_df = auth_df.iloc[:SMOKE_TEST_N].reset_index(drop=True)

    effective_batch = PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
    effective_epochs = MAX_STEPS * effective_batch / len(synth_df)
    print(f"Compute budget: {MAX_STEPS} step x {effective_batch} (effective batch) / "
          f"{len(synth_df)} file sintetis \u2248 {effective_epochs:.1f} epoch efektif")

    processor = WhisperProcessor.from_pretrained(MODEL_NAME)  # meniru zero-shot: tanpa language=/task= saat load
    forced_decoder_ids = processor.get_decoder_prompt_ids(language=LANGUAGE, task=TASK)

    pred_out_path = os.path.join(OUTPUT_DIR, "synthetic_only_whisper_predictions.csv")

    # ---- CEK APAKAH SUDAH SELESAI TOTAL (resume: skip training+eval sepenuhnya) ----
    if os.path.exists(pred_out_path) and os.path.exists(FINAL_MODEL_DIR):
        print(f"Model final & prediksi sudah ada -- SKIP total (resume).\n"
              f"  Model   : {FINAL_MODEL_DIR}\n  Prediksi: {pred_out_path}")
        return pd.read_csv(pred_out_path)

    # ---- TRAINING (dgn resume dari checkpoint kalau ada) ----
    if os.path.exists(FINAL_MODEL_DIR):
        print(f"Model final sudah tersimpan di {FINAL_MODEL_DIR} -- muat model itu, skip training.")
        model = WhisperForConditionalGeneration.from_pretrained(FINAL_MODEL_DIR)
        model.to(device)
    else:
        print(f"\n{'=' * 60}\nTRAINING -- {len(synth_df)} file sintetis, TANPA k-fold\n{'=' * 60}")
        print("Membangun dataset training...")
        train_ds = build_synthetic_dataset(synth_df, processor)

        model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
        model.config.forced_decoder_ids = None
        model.config.suppress_tokens = []
        model.to(device)

        # Verifikasi eksplisit: full fine-tuning = 100% parameter trainable
        # (JANGAN cuma percaya, buktikan lewat print -- pola konsisten sepanjang
        # proyek ini, sama seperti cek trainable_parameters di MMS/SeamlessM4T).
        n_total = sum(p.numel() for p in model.parameters())
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Verifikasi parameter: {n_trainable:,}/{n_total:,} trainable "
              f"({100 * n_trainable / n_total:.1f}%) -- full fine-tuning, harus 100%")
        assert n_trainable == n_total, "BUKAN full fine-tuning -- ada parameter yang kebeku, cek model!"

        data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

        training_args = Seq2SeqTrainingArguments(
            output_dir=OUTPUT_DIR,
            per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            learning_rate=LEARNING_RATE,
            warmup_steps=WARMUP_STEPS,
            max_steps=MAX_STEPS,
            gradient_checkpointing=GRADIENT_CHECKPOINTING,
            fp16=FP16,
            logging_steps=LOGGING_STEPS,
            save_strategy="steps",
            save_steps=SAVE_STEPS,
            save_total_limit=SAVE_TOTAL_LIMIT,
            eval_strategy="no",     # test set TIDAK disentuh sama sekali selama training
            report_to=[],
            seed=RANDOM_SEED,
            predict_with_generate=False,
            remove_unused_columns=False,
        )

        trainer_kwargs = dict(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            data_collator=data_collator,
        )
        # transformers >=4.46-ish rename tokenizer= -> processing_class= di Trainer/
        # Seq2SeqTrainer -- dicek dinamis via signature, sama pola dgn skrip Authentic-Only.
        sig_params = inspect.signature(Seq2SeqTrainer.__init__).parameters
        if "processing_class" in sig_params:
            trainer_kwargs["processing_class"] = processor.feature_extractor
        elif "tokenizer" in sig_params:
            trainer_kwargs["tokenizer"] = processor.feature_extractor

        trainer = Seq2SeqTrainer(**trainer_kwargs)

        # Resume dari checkpoint terakhir kalau OUTPUT_DIR/checkpoint-* sudah ada
        # (mis. Colab disconnect di tengah 600 step) -- beda dari Authentic-Only yang
        # resume-nya per-fold (fold selesai/belum); di sini per-checkpoint di dalam
        # 1 training run panjang.
        existing_checkpoints = sorted(Path(OUTPUT_DIR).glob("checkpoint-*"))
        resume = bool(existing_checkpoints)
        if resume:
            print(f"Checkpoint lama ditemukan ({len(existing_checkpoints)}) -- resume training dari checkpoint terakhir.")
        trainer.train(resume_from_checkpoint=resume)

        log_history = trainer.state.log_history
        Path(os.path.join(OUTPUT_DIR, "train_log_history.json")).write_text(
            json.dumps(log_history, indent=2), encoding="utf-8"
        )

        # SIMPAN MODEL FINAL -- beda dari Authentic-Only (model dibuang tiap fold).
        # Model Synthetic-Only BISA dipakai lagi (cuma 1 model per arsitektur).
        os.makedirs(FINAL_MODEL_DIR, exist_ok=True)
        model.save_pretrained(FINAL_MODEL_DIR)
        processor.save_pretrained(FINAL_MODEL_DIR)
        print(f"Model final tersimpan ke: {FINAL_MODEL_DIR}")

    # ---- EVALUASI: prediksi ke SEMUA segmen authentic (TIDAK ADA fold) ----
    if os.path.exists(pred_out_path):
        print(f"Prediksi sudah ada di {pred_out_path} -- skip eval (resume).")
        pred_df = pd.read_csv(pred_out_path)
    else:
        print(f"\n{'=' * 60}\nEVALUASI -- {len(auth_df)} segmen authentic\n{'=' * 60}")
        predictions = predict_all(model, processor, forced_decoder_ids, auth_df, device)
        pred_df = pd.DataFrame(predictions).sort_values("file_id").reset_index(drop=True)

        assert pred_df.file_id.nunique() == len(pred_df), "Ada file_id terprediksi LEBIH dari 1x!"
        if len(pred_df) == len(auth_df):
            print(f"\nSanity check akhir: LOLOS ({len(pred_df)}/{len(auth_df)} segmen terprediksi tepat 1x)")
        else:
            print(f"\n\u26a0 PERINGATAN: cuma {len(pred_df)}/{len(auth_df)} segmen berhasil diprediksi "
                  f"(sisanya di-skip krn error, lihat log GAGAL di atas). Cek sebelum lanjut ke evaluasi CI.")

        pred_df.to_csv(pred_out_path, index=False)
        print(f"Tersimpan: {pred_out_path}")

    del model
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return pred_df


if __name__ == "__main__":
    main()
