"""
finetune_whisper_combined_colab.py

Tujuan
------
Fine-tuning Whisper-small ke kondisi Combined: K-fold CV group-aware SAMA
PERSIS skema Authentic-Only (5 fold, fold_authentic disisihkan tiap
putaran), TAPI training set tiap fold = (authentic_train_fold) + (SEMUA 504
file sintetis) digabung jadi satu training set. Evaluasi ke fold authentic
yang disisihkan (rotasi), persis Authentic-Only -- BUKAN ke semua 106
sekaligus seperti Synthetic-Only.

Sumber skema: rangkuman_metodologi_update_2.md Bagian 4.2 -- "Combined | 106
authentic + 504 sintetis | K-fold CV (fold disisihkan dari training tiap
putaran)". Tidak ada rasio/oversampling/pembobotan yang disebut secara
eksplisit di metodologi -- skrip ini GABUNG APA ADANYA (concatenate polos),
bukan oversampling authentic atau downsampling sintetis. Kalau nanti mau
diuji rasio campuran lain, itu perubahan desain yang perlu didiskusikan
dulu, bukan diasumsikan di sini.

============================================================================
PERBEDAAN STRUKTURAL vs finetune_whisper_authentic_colab.py
============================================================================
- Training set tiap fold DITAMBAH 504 file sintetis (authentic_train_fold
  TETAP dirotasi per fold spt biasa, sintetisnya SAMA & PENUH di semua fold
  -- tidak ikut dirotasi/displit, karena sintetis bukan bagian dari skema
  k-fold, cuma "ditumpangkan").
- Model TETAP dibuang tiap fold (save_strategy="no", SAMA PERSIS
  Authentic-Only) -- BEDA dari Synthetic-Only yang modelnya disimpan,
  karena di sini tetap ada 5 model per fold (bukan 1 model tunggal).
- OPTIMISASI: bagian sintetis (504 file) di-preprocess (ekstraksi fitur
  audio) CUMA SEKALI di luar loop fold, bukan 5x ulang per fold -- karena
  isinya identik di semua fold, cuma bagian authentic yang beda tiap fold.
  Ini menghemat ~4x waktu preprocessing audio sintetis dibanding pendekatan
  naif (proses ulang 504 file tiap fold). Digabung via
  datasets.concatenate_datasets() tiap fold, bukan diproses ulang.
- Resume-capable: skip fold yang predictions_fold{N}.csv-nya sudah ada --
  SAMA PERSIS pola Authentic-Only (BUKAN pola resume checkpoint
  Synthetic-Only, karena di sini tetap ada 5 run terpisah per fold, bukan
  1 run panjang).

============================================================================
KONSISTENSI HYPERPARAMETER (WAJIB SAMA PERSIS dgn Authentic-Only & Synthetic-Only Whisper)
============================================================================
MAX_STEPS=600, learning_rate=1e-5, warmup_steps=60, per_device_batch=8,
grad_accum=2 (effective 16), fp16=True, gradient_checkpointing=True --
disalin PERSIS (variabel kontrol WAJIB sama across ketiga kondisi utk model
yang sama, lihat ringkasan_konfigurasi_fine_tuning.md & rangkuman metodologi
10.2: "WAJIB dipakai ulang persis di Synthetic-Only & Combined, per model").

Implikasi epoch efektif per fold: training set ~85 authentic + 504 sintetis
= ~589 contoh (bervariasi tipis antar fold krn ukuran fold authentic tidak
identik). 600 step x 16 (effective batch) / ~589 ~ 16.3 epoch efektif --
di antara Authentic-Only (~113 epoch, cuma ~85 contoh) dan Synthetic-Only
(~19 epoch, 504 contoh), lebih dekat ke Synthetic-Only krn sintetis
mendominasi proporsi data (~85,6% dari total). Dicetak per fold di runtime
(ukuran fold authentic sedikit bervariasi).

Decoding evaluasi: SAMA PERSIS Authentic-Only/Synthetic-Only/Zero-Shot
(greedy, language="id", task="transcribe", tanpa max_length/num_beams).

============================================================================
CATATAN -- FILE FOLD & PATH AUDIO
============================================================================
MANIFEST_WITH_FOLD_PATH = authentic_manifest_with_fold.csv, file YANG SAMA
PERSIS dipakai finetune_whisper_authentic_colab.py -- JANGAN generate ulang
k-fold, pakai fold assignment yang sudah ada (dari kfold_split_authentic_colab.py,
sudah dipakai & terbukti jalan di Authentic-Only).

Path audio sintetis (SYNTHETIC_AUDIO_DIR, SYNTHETIC_AUDIO_EXT) SAMA persis
dgn finetune_whisper_synthetic_colab.py -- DIKONFIRMASI user (folder
bugis_tts/audio, flat, ekstensi .mp3).

Prasyarat sebelum run di Colab
-------------------------------
- Google Drive sudah di-mount.
- authentic_manifest_with_fold.csv, synthetic_manifest_cleaned.csv ada di
  path CONFIG.
- 106 file audio authentic + 504 file audio sintetis sudah ter-mount.
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

from datasets import Dataset, concatenate_datasets
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    set_seed,
)

# ---- EDIT BAGIAN INI ----
MANIFEST_WITH_FOLD_PATH = "/content/drive/MyDrive/bugis_authentic/authentic_manifest_with_fold.csv"   # SAMA PERSIS file Authentic-Only, JANGAN generate ulang
AUDIO_DIR = "/content/drive/MyDrive/bugis_authentic/audio"    # SAMA dgn Authentic-Only/zero-shot

SYNTHETIC_MANIFEST_PATH = "/content/drive/MyDrive/bugis_authentic/synthetic_manifest_cleaned.csv"   # cek lokasi persis -- lihat catatan sesi Synthetic-Only
SYNTHETIC_AUDIO_DIR = "/content/drive/MyDrive/bugis_tts/audio"   # DIKONFIRMASI user
SYNTHETIC_AUDIO_EXT = ".mp3"    # DIKONFIRMASI user

OUTPUT_DIR = "/content/drive/MyDrive/bugis_authentic/combined_results"

MODEL_NAME = "openai/whisper-small"
LANGUAGE = "id"
TASK = "transcribe"

K_FOLDS = 5                  # HARUS sama dgn kfold_split_authentic_colab.py
RANDOM_SEED = 42

# ---- Compute budget: SAMA PERSIS dgn Authentic-Only & Synthetic-Only Whisper (variabel kontrol) ----
MAX_STEPS = 600
LEARNING_RATE = 1e-5
WARMUP_STEPS = 60
PER_DEVICE_TRAIN_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 2     # effective batch = 16
LOGGING_STEPS = 20

FP16 = True
GRADIENT_CHECKPOINTING = True

SMOKE_TEST_N = None          # set angka kecil (mis. 3) utk smoke test dulu -- SANGAT disarankan
                              # sebelum full run 5 fold (mahal, ~1 jam/fold spt Authentic-Only)
# --------------------------


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """SAMA PERSIS dgn finetune_whisper_authentic_colab.py / finetune_whisper_synthetic_colab.py."""
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
    """SAMA PERSIS dgn skrip lain di pipeline ini."""
    text = unicodedata.normalize("NFC", text.strip())
    return " ".join(text.split())


def load_audio(path: str) -> np.ndarray:
    array, _sr = librosa.load(path, sr=16000, mono=True)
    return array


def synthetic_audio_path(file_id: str) -> str:
    """SAMA PERSIS dgn helper di finetune_whisper_synthetic_colab.py."""
    return os.path.join(SYNTHETIC_AUDIO_DIR, file_id + SYNTHETIC_AUDIO_EXT)


def build_authentic_dataset(df: pd.DataFrame, processor: WhisperProcessor) -> Dataset:
    """Komponen authentic (train fold) -- dipanggil ULANG tiap fold, karena
    isinya beda tiap fold (rotasi).

    CATATAN BUG YANG SUDAH DIPERBAIKI: file_id di authentic_manifest itu
    penomoran angka murni (1-107, lihat rangkuman metodologi 4.2), jadi
    pandas otomatis infer dtype int64 -- sedangkan file_id sintetis
    ("ardi001") ke-infer sbg string. concatenate_datasets() menolak gabung
    dua Dataset dgn tipe kolom beda (int64 vs string) utk kolom yang sama.
    FIX: paksa file_id jadi string SEBELUM Dataset.from_pandas(), di KEDUA
    builder (authentic & synthetic) -- supaya skema selalu cocok, apa pun
    isi datanya."""
    df = df.copy()
    df["file_id"] = df["file_id"].astype(str)
    ds = Dataset.from_pandas(df[["file_id", "text", "audio_filename"]].reset_index(drop=True))

    def _prepare(example):
        audio_array = load_audio(os.path.join(AUDIO_DIR, example["audio_filename"]))
        example["input_features"] = processor.feature_extractor(
            audio_array, sampling_rate=16000
        ).input_features[0]
        example["labels"] = processor.tokenizer(example["text"]).input_ids
        return example

    ds = ds.map(_prepare, remove_columns=["text", "audio_filename"])
    return ds


def build_synthetic_dataset(df: pd.DataFrame, processor: WhisperProcessor) -> Dataset:
    """Komponen sintetis (SEMUA 504) -- dipanggil CUMA SEKALI di main(), di
    luar loop fold (lihat catatan OPTIMISASI di docstring atas). file_id
    dipaksa string juga di sini (defensif, konsisten dgn fix di
    build_authentic_dataset -- lihat catatan bug di sana)."""
    df = df.copy()
    df["file_id"] = df["file_id"].astype(str)
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
    """SAMA PERSIS dgn skrip lain di pipeline ini."""
    audio, _ = librosa.load(str(audio_path), sr=16000)
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features
    if device == "cuda":
        input_features = input_features.to("cuda")
    with torch.no_grad():
        predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
    return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]


def predict_fold(model, processor, forced_decoder_ids, test_df: pd.DataFrame, fold: int, device) -> List[Dict]:
    """SAMA PERSIS pola predict_fold() di finetune_whisper_authentic_colab.py."""
    model.eval()
    results = []
    rows = test_df.reset_index(drop=True)

    for row in rows.itertuples():
        audio_path = os.path.join(AUDIO_DIR, row.audio_filename)
        reference = normalize(row.text)
        try:
            hypothesis = normalize(
                transcribe_whisper_one(model, processor, forced_decoder_ids, audio_path, device)
            )
        except Exception as e:
            print(f"  \u26a0 GAGAL file_id={row.file_id} (fold {fold}): {e}")
            continue

        results.append({
            "file_id": row.file_id,
            "recording_id": row.recording_id,
            "fold": fold,
            "reference": reference,
            "hypothesis": hypothesis,
        })
        print(f"  [fold {fold}] file_id={row.file_id}: ref='{reference[:60]}' | hyp='{hypothesis[:60]}'")

    return results


def main():
    set_seed(RANDOM_SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("PERINGATAN: GPU tidak terdeteksi -- training di CPU akan SANGAT lambat. "
              "Pastikan runtime Colab diset ke GPU.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(MANIFEST_WITH_FOLD_PATH)
    assert len(df) == 106, f"Expected 106 baris authentic, dapat {len(df)}"
    assert set(df.fold.unique()) == set(range(K_FOLDS)), \
        f"fold di manifest ({sorted(df.fold.unique())}) tidak cocok K_FOLDS={K_FOLDS}"

    synth_df = pd.read_csv(SYNTHETIC_MANIFEST_PATH)
    assert len(synth_df) == 504, f"Expected 504 baris sintetis, dapat {len(synth_df)} -- cek SYNTHETIC_MANIFEST_PATH"

    if SMOKE_TEST_N:
        print(f"\u26a0 MODE SMOKE TEST -- train authentic per fold & sintetis dipotong ke {SMOKE_TEST_N} "
              f"baris, test per fold juga dipotong. Set SMOKE_TEST_N = None utk full run.\n"
              f"\u26a0 INGAT: pakai OUTPUT_DIR terpisah dari full run.\n")
        synth_df = synth_df.iloc[:SMOKE_TEST_N].reset_index(drop=True)

    processor = WhisperProcessor.from_pretrained(MODEL_NAME)
    forced_decoder_ids = processor.get_decoder_prompt_ids(language=LANGUAGE, task=TASK)

    pred_out_path = os.path.join(OUTPUT_DIR, "combined_whisper_predictions.csv")
    if os.path.exists(pred_out_path):
        print(f"Prediksi gabungan semua fold sudah ada -- SKIP total (resume): {pred_out_path}")
        return pd.read_csv(pred_out_path)

    # Cek dulu fold mana yang BELUM selesai -- kalau semua sudah, tidak perlu
    # bangun dataset sintetis sama sekali (hemat waktu preprocessing).
    fold_pred_paths = {
        fold: os.path.join(OUTPUT_DIR, f"predictions_combined_fold{fold}.csv")
        for fold in range(K_FOLDS)
    }
    folds_needed = [f for f in range(K_FOLDS) if not os.path.exists(fold_pred_paths[f])]

    synth_ds = None
    if folds_needed:
        print(f"Membangun komponen sintetis ({len(synth_df)} file) -- SEKALI SAJA, dipakai ulang di "
              f"{len(folds_needed)} fold yang belum selesai ({folds_needed})...")
        synth_ds = build_synthetic_dataset(synth_df, processor)
    else:
        print("Semua fold sudah selesai (resume) -- skip bangun komponen sintetis, langsung agregasi.")

    all_predictions: List[Dict] = []
    all_log_histories: Dict[str, Any] = {}

    for fold in range(K_FOLDS):
        fold_pred_path = fold_pred_paths[fold]

        if os.path.exists(fold_pred_path):
            print(f"\n{'=' * 60}\nFOLD {fold + 1}/{K_FOLDS} -- SUDAH ADA, DI-SKIP (resume)\n{'=' * 60}")
            fold_pred_df = pd.read_csv(fold_pred_path)
            all_predictions.extend(fold_pred_df.to_dict("records"))
            fold_log_path = os.path.join(OUTPUT_DIR, f"log_history_combined_fold{fold}.json")
            if os.path.exists(fold_log_path):
                all_log_histories[f"fold{fold}"] = json.loads(Path(fold_log_path).read_text(encoding="utf-8"))
            print(f"Dimuat dari: {fold_pred_path} ({len(fold_pred_df)} baris)")
            continue

        print(f"\n{'=' * 60}\nFOLD {fold + 1}/{K_FOLDS}\n{'=' * 60}")
        auth_train_df = df[df.fold != fold].reset_index(drop=True)
        test_df = df[df.fold == fold].reset_index(drop=True)

        if SMOKE_TEST_N:
            auth_train_df = auth_train_df.iloc[:SMOKE_TEST_N].reset_index(drop=True)
            test_df = test_df.iloc[:SMOKE_TEST_N].reset_index(drop=True)

        print(f"Train: {len(auth_train_df)} segmen authentic dari {auth_train_df.recording_id.nunique()} rekaman "
              f"+ {len(synth_df)} file sintetis = {len(auth_train_df) + len(synth_df)} total")
        print(f"Test : {len(test_df)} segmen dari {test_df.recording_id.nunique()} rekaman "
              f"(TIDAK overlap dengan rekaman train authentic -- group-aware; sintetis tidak relevan "
              f"utk cek ini krn sumbernya beda total)")

        assert set(auth_train_df.recording_id) & set(test_df.recording_id) == set(), \
            f"BOCOR di fold {fold}: ada recording_id yang muncul di train DAN test!"

        effective_batch = PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
        total_train = len(auth_train_df) + len(synth_df)
        effective_epochs = MAX_STEPS * effective_batch / total_train
        print(f"Compute budget: {MAX_STEPS} step x {effective_batch} (effective batch) / "
              f"{total_train} contoh \u2248 {effective_epochs:.1f} epoch efektif")

        auth_ds = build_authentic_dataset(auth_train_df, processor)
        train_ds = concatenate_datasets([synth_ds, auth_ds]).shuffle(seed=RANDOM_SEED)

        model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
        model.config.forced_decoder_ids = None
        model.config.suppress_tokens = []
        model.to(device)

        n_total = sum(p.numel() for p in model.parameters())
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Verifikasi parameter: {n_trainable:,}/{n_total:,} trainable "
              f"({100 * n_trainable / n_total:.1f}%) -- full fine-tuning, harus 100%")
        assert n_trainable == n_total, "BUKAN full fine-tuning -- ada parameter yang kebeku, cek model!"

        data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

        fold_dir = os.path.join(OUTPUT_DIR, f"whisper_fold{fold}")
        training_args = Seq2SeqTrainingArguments(
            output_dir=fold_dir,
            per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            learning_rate=LEARNING_RATE,
            warmup_steps=WARMUP_STEPS,
            max_steps=MAX_STEPS,
            gradient_checkpointing=GRADIENT_CHECKPOINTING,
            fp16=FP16,
            logging_steps=LOGGING_STEPS,
            save_strategy="no",     # model DIBUANG tiap fold -- SAMA PERSIS Authentic-Only
            eval_strategy="no",
            report_to=[],
            seed=RANDOM_SEED,
            predict_with_generate=False,
            remove_unused_columns=False,
        )

        trainer_kwargs = dict(
            model=model, args=training_args, train_dataset=train_ds, data_collator=data_collator,
        )
        sig_params = inspect.signature(Seq2SeqTrainer.__init__).parameters
        if "processing_class" in sig_params:
            trainer_kwargs["processing_class"] = processor.feature_extractor
        elif "tokenizer" in sig_params:
            trainer_kwargs["tokenizer"] = processor.feature_extractor

        trainer = Seq2SeqTrainer(**trainer_kwargs)
        trainer.train()
        fold_log_history = trainer.state.log_history
        all_log_histories[f"fold{fold}"] = fold_log_history

        fold_predictions = predict_fold(model, processor, forced_decoder_ids, test_df, fold, device)
        all_predictions.extend(fold_predictions)

        pd.DataFrame(fold_predictions).to_csv(fold_pred_path, index=False)
        fold_log_path = os.path.join(OUTPUT_DIR, f"log_history_combined_fold{fold}.json")
        Path(fold_log_path).write_text(json.dumps(fold_log_history, indent=2), encoding="utf-8")
        print(f"Fold {fold}: {len(fold_predictions)} prediksi tersimpan ke {fold_pred_path}")

        del model, trainer
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    pred_df = pd.DataFrame(all_predictions).sort_values("file_id").reset_index(drop=True)
    assert pred_df.file_id.nunique() == len(pred_df), "Ada file_id yang terprediksi LEBIH dari 1x -- cek logika fold!"
    if len(pred_df) == 106:
        print(f"\nSanity check akhir: LOLOS (106/106 segmen terprediksi tepat 1x lewat skema k-fold)")
    else:
        print(f"\n\u26a0 PERINGATAN: cuma {len(pred_df)}/106 segmen berhasil diprediksi "
              f"(sisanya di-skip krn error, lihat log GAGAL di atas). Cek sebelum lanjut ke evaluasi.")

    pred_df.to_csv(pred_out_path, index=False)
    print(f"Tersimpan: {pred_out_path}")

    log_out_path = os.path.join(OUTPUT_DIR, "train_log_history_all_folds_combined_whisper.json")
    with open(log_out_path, "w") as f:
        json.dump(all_log_histories, f, indent=2)
    print(f"Tersimpan: {log_out_path}")

    return pred_df


if __name__ == "__main__":
    main()
