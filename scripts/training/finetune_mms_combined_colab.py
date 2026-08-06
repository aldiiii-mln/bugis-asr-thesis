"""
finetune_mms_combined_colab.py

Tujuan
------
Fine-tuning MMS (facebook/mms-1b-all, adapter 'ind', adapter-only) ke
kondisi Combined: K-fold CV group-aware SAMA PERSIS skema Authentic-Only (5
fold), TAPI training set tiap fold = (authentic_train_fold) + (SEMUA 504
file sintetis) digabung. Evaluasi ke fold authentic yang disisihkan
(rotasi). Pola strukturalnya identik dengan finetune_whisper_combined_colab.py
-- baca docstring skrip itu dulu kalau belum (termasuk soal skema Combined
dari metodologi 4.2 & keputusan "gabung apa adanya, tanpa oversampling").
Penjelasan di sini fokus ke hal yang KHUSUS MMS saja.

============================================================================
FIX YANG SUDAH DIPASANG DARI AWAL (bukan nunggu error lagi)
============================================================================
finetune_whisper_combined_colab.py sempat error saat pertama dites:
concatenate_datasets() menolak gabung dataset authentic (file_id ke-infer
int64, krn isinya cuma angka 1-107) dengan dataset sintetis (file_id
ke-infer string, krn isinya "ardi001" dst). FIX itu (paksa file_id jadi
str() SEBELUM Dataset.from_pandas(), di KEDUA builder) sudah dipasang dari
awal di build_authentic_dataset() & build_synthetic_dataset() di bawah --
bukan menunggu error yang sama muncul lagi di sini.

============================================================================
PERBEDAAN STRUKTURAL vs finetune_mms_authentic_colab.py
============================================================================
- Training set tiap fold DITAMBAH 504 file sintetis (SAMA PERSIS pola
  Whisper Combined -- authentic_train_fold dirotasi, sintetis PENUH & SAMA
  di semua fold, tidak ikut dirotasi).
- Model TETAP dibuang tiap fold (save_strategy="no") -- SAMA Authentic-Only
  & Whisper Combined. Konsekuensinya: TIDAK ADA isu ukuran checkpoint besar
  spt di MMS Synthetic-Only (yang modelnya disimpan) -- di sini model
  memang tidak pernah disimpan sama sekali, jadi tidak relevan.
- OPTIMISASI: komponen sintetis (504 file) diproses fitur audionya CUMA
  SEKALI di luar loop fold (SAMA PERSIS alasan & pola Whisper Combined).
- Resume-capable: skip fold yang predictions sudah ada -- pola Authentic-Only,
  BUKAN pola resume checkpoint Synthetic-Only.

============================================================================
CEK VOCABULARY (SAMA seperti MMS Synthetic-Only, BUKAN seperti Authentic-Only)
============================================================================
Combined training text = authentic (106, semua muncul sbg train di suatu
fold) + sintetis (504) -- dua himpunan berbeda, jadi dicek TERPISAH (label
"sintetis" & "authentic"), SAMA PERSIS alasan di finetune_mms_synthetic_colab.py.
Dicek SEKALI di awal main() (bukan per fold -- teksnya tidak berubah antar
fold, cuma pembagian train/test-nya yang beda).

============================================================================
KONSISTENSI HYPERPARAMETER (WAJIB SAMA PERSIS dgn Authentic-Only & Synthetic-Only MMS)
============================================================================
MAX_STEPS=150, learning_rate=1e-3, warmup_steps=20, per_device_batch=4,
grad_accum=8 (effective 32), group_by_length=True, dropout semua 0,
ctc_loss_reduction="mean", load_adapter('ind') -- disalin PERSIS (variabel
kontrol WAJIB sama across ketiga kondisi utk model yang sama).

Implikasi epoch efektif per fold: ~85 authentic + 504 sintetis = ~589 contoh
(bervariasi tipis antar fold). 150 step x 32 (effective batch) / ~589 ~ 8.1
epoch efektif -- dicetak per fold di runtime.

Decoding evaluasi: greedy argmax (TANPA beam search) -- SAMA PERSIS
Authentic-Only/Synthetic-Only/Zero-Shot.

============================================================================
CATATAN -- FILE FOLD & PATH AUDIO (SAMA dgn Whisper Combined)
============================================================================
MANIFEST_WITH_FOLD_PATH = authentic_manifest_with_fold.csv, file YANG SAMA
PERSIS dipakai finetune_mms_authentic_colab.py & finetune_whisper_combined_colab.py
-- JANGAN generate ulang k-fold.

Path audio sintetis (SYNTHETIC_AUDIO_DIR, SYNTHETIC_AUDIO_EXT) SAMA persis
dgn skrip Synthetic-Only -- DIKONFIRMASI user (folder bugis_tts/audio,
flat, ekstensi .mp3).

Prasyarat sebelum run di Colab
-------------------------------
- Google Drive sudah di-mount.
- authentic_manifest_with_fold.csv, synthetic_manifest_cleaned.csv ada di
  path CONFIG.
- 106 file audio authentic + 504 file audio sintetis sudah ter-mount.
- pip install: transformers datasets accelerate librosa soundfile

Gaya skrip: Colab (tanpa argparse), konsisten dgn skrip lain di pipeline ini.
"""

import os
import gc
import json
import inspect
import dataclasses
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
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    TrainingArguments,
    Trainer,
    set_seed,
)

# ---- EDIT BAGIAN INI ----
MANIFEST_WITH_FOLD_PATH = "/content/drive/MyDrive/bugis_authentic/authentic_manifest_with_fold.csv"   # SAMA PERSIS file Authentic-Only/Whisper Combined
AUDIO_DIR = "/content/drive/MyDrive/bugis_authentic/audio"

SYNTHETIC_MANIFEST_PATH = "/content/drive/MyDrive/bugis_authentic/synthetic_manifest_cleaned.csv"   # cek lokasi persis
SYNTHETIC_AUDIO_DIR = "/content/drive/MyDrive/bugis_tts/audio"   # DIKONFIRMASI user
SYNTHETIC_AUDIO_EXT = ".mp3"    # DIKONFIRMASI user

OUTPUT_DIR = "/content/drive/MyDrive/bugis_authentic/combined_results_mms"   # folder TERPISAH dari Whisper Combined (combined_results), sesuai permintaan user

MMS_CHECKPOINT = "facebook/mms-1b-all"
MMS_TARGET_LANG = "ind"

K_FOLDS = 5
RANDOM_SEED = 42

# ---- Compute budget: SAMA PERSIS dgn Authentic-Only & Synthetic-Only MMS (variabel kontrol) ----
MAX_STEPS = 150
LEARNING_RATE = 1e-3
WARMUP_STEPS = 20
PER_DEVICE_TRAIN_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 8     # effective batch = 32
USE_GROUP_BY_LENGTH = True

LOGGING_STEPS = 10

SMOKE_TEST_N = None          # set angka kecil (mis. 3) utk smoke test dulu -- SANGAT disarankan
                              # sebelum full run 5 fold
# --------------------------


def normalize(text: str) -> str:
    """SAMA PERSIS dgn skrip lain di pipeline ini."""
    text = unicodedata.normalize("NFC", text.strip())
    return " ".join(text.split())


def load_audio(path: str) -> np.ndarray:
    array, _sr = librosa.load(path, sr=16000, mono=True)
    return array


def synthetic_audio_path(file_id: str) -> str:
    """SAMA PERSIS dgn helper di skrip Synthetic-Only/Whisper Combined."""
    return os.path.join(SYNTHETIC_AUDIO_DIR, file_id + SYNTHETIC_AUDIO_EXT)


def check_vocab_coverage(processor: Wav2Vec2Processor, texts: List[str], label: str) -> None:
    """SAMA PERSIS dgn finetune_mms_synthetic_colab.py."""
    vocab = set(processor.tokenizer.get_vocab().keys())
    all_chars = set("".join(texts)) - {" "}
    missing = sorted(
        c for c in all_chars
        if c not in vocab and c.upper() not in vocab and c.lower() not in vocab
    )
    print(f"\n--- Cek cakupan vocabulary MMS ('{MMS_TARGET_LANG}') -- {label} ---")
    print(f"Ukuran vocab: {len(vocab)} token")
    if missing:
        print(f"\u26a0 PERINGATAN: {len(missing)} karakter di teks {label} TIDAK ditemukan di vocab: {missing}")
        print("  Karakter ini TIDAK PERNAH bisa muncul di prediksi MMS -- WAJIB dicatat di Limitasi tesis.")
    else:
        print(f"Semua karakter di teks {label} ADA di vocab -- aman.")
    print("---\n")


def verify_model_config(model, processor) -> bool:
    """SAMA PERSIS dgn finetune_mms_authentic_colab.py / finetune_mms_synthetic_colab.py."""
    print("\n--- Verifikasi config model ---")
    expected = {
        "attention_dropout": 0.0, "hidden_dropout": 0.0,
        "feat_proj_dropout": 0.0, "layerdrop": 0.0,
        "ctc_loss_reduction": "mean",
    }
    all_ok = True
    for key, exp in expected.items():
        actual = getattr(model.config, key, "<<ATTRIBUTE TIDAK ADA>>")
        ok = (actual == exp)
        all_ok = all_ok and ok
        print(f"  config.{key}: diminta={exp}, aktual={actual}  {'OK' if ok else '<<< MISMATCH!'}")

    vocab_size = len(processor.tokenizer)
    lm_head_out = model.lm_head.out_features
    vocab_ok = (vocab_size == lm_head_out)
    all_ok = all_ok and vocab_ok
    print(f"  vocab tokenizer ({vocab_size}) vs dimensi output lm_head ({lm_head_out})  "
          f"{'OK' if vocab_ok else '<<< MISMATCH!'}")

    if not all_ok:
        print("\u26a0\u26a0\u26a0 PERINGATAN KERAS: ada ketidakcocokan -- JANGAN lanjut training.")
    else:
        print("Semua cocok -- aman lanjut.")
    print("---\n")
    return all_ok


def freeze_for_adapter_only(model: Wav2Vec2ForCTC) -> None:
    """SAMA PERSIS dgn finetune_mms_authentic_colab.py / finetune_mms_synthetic_colab.py."""
    model.freeze_feature_encoder()
    model.freeze_base_model()

    adapter_weights = model._get_adapters()
    n_trainable = 0
    for param in adapter_weights.values():
        param.requires_grad = True
        n_trainable += param.numel()

    n_total = sum(p.numel() for p in model.parameters())
    pct = 100 * n_trainable / n_total if n_total else 0
    print(f"Adapter-only fine-tuning: {n_trainable:,} / {n_total:,} parameter trainable ({pct:.2f}%)")
    if n_trainable == 0:
        print("\u26a0\u26a0\u26a0 PERINGATAN KERAS: 0 parameter trainable! JANGAN lanjut training.")


@dataclass
class DataCollatorCTCWithPadding:
    """SAMA PERSIS dgn finetune_mms_authentic_colab.py / finetune_mms_synthetic_colab.py."""
    processor: Any
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        batch = self.processor.pad(input_features, padding=self.padding, return_tensors="pt")
        labels_batch = self.processor.pad(labels=label_features, padding=self.padding, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch


def build_authentic_dataset(df: pd.DataFrame, processor: Wav2Vec2Processor) -> Dataset:
    """Komponen authentic (train fold) -- dipanggil ULANG tiap fold.

    file_id DIPAKSA STRING sebelum Dataset.from_pandas() -- lihat catatan
    FIX di docstring atas (bug yang sudah kejadian di Whisper Combined)."""
    df = df.copy()
    df["file_id"] = df["file_id"].astype(str)
    ds = Dataset.from_pandas(df[["file_id", "text", "audio_filename"]].reset_index(drop=True))

    def _prepare(example):
        audio_array = load_audio(os.path.join(AUDIO_DIR, example["audio_filename"]))
        example["input_values"] = processor.feature_extractor(
            audio_array, sampling_rate=16000
        ).input_values[0]
        example["input_length"] = len(example["input_values"])
        example["labels"] = processor.tokenizer(example["text"]).input_ids
        return example

    ds = ds.map(_prepare, remove_columns=["text", "audio_filename"])
    return ds


def build_synthetic_dataset(df: pd.DataFrame, processor: Wav2Vec2Processor) -> Dataset:
    """Komponen sintetis (SEMUA 504) -- dipanggil CUMA SEKALI di main().
    file_id dipaksa string juga (defensif, konsisten dgn build_authentic_dataset)."""
    df = df.copy()
    df["file_id"] = df["file_id"].astype(str)
    ds = Dataset.from_pandas(df[["file_id", "text"]].reset_index(drop=True))

    def _prepare(example):
        audio_array = load_audio(synthetic_audio_path(example["file_id"]))
        example["input_values"] = processor.feature_extractor(
            audio_array, sampling_rate=16000
        ).input_values[0]
        example["input_length"] = len(example["input_values"])
        example["labels"] = processor.tokenizer(example["text"]).input_ids
        return example

    ds = ds.map(_prepare, remove_columns=["text"])
    return ds


def transcribe_mms_one(model, processor, audio_path, device) -> str:
    """SAMA PERSIS dgn skrip lain di pipeline ini -- greedy argmax."""
    audio, _ = librosa.load(str(audio_path), sr=16000)
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    input_values = inputs.input_values
    if device == "cuda":
        input_values = input_values.to("cuda")
    with torch.no_grad():
        logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    return processor.batch_decode(predicted_ids)[0]


def predict_fold(model, processor, test_df: pd.DataFrame, fold: int, device) -> List[Dict]:
    """SAMA PERSIS pola predict_fold() di finetune_whisper_combined_colab.py."""
    model.eval()
    results = []
    rows = test_df.reset_index(drop=True)

    for row in rows.itertuples():
        audio_path = os.path.join(AUDIO_DIR, row.audio_filename)
        reference = normalize(row.text)
        try:
            hypothesis = normalize(transcribe_mms_one(model, processor, audio_path, device))
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
        print("PERINGATAN: GPU tidak terdeteksi -- MMS-1b di CPU akan SANGAT lambat.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(MANIFEST_WITH_FOLD_PATH)
    assert len(df) == 106, f"Expected 106 baris authentic, dapat {len(df)}"
    assert set(df.fold.unique()) == set(range(K_FOLDS)), \
        f"fold di manifest ({sorted(df.fold.unique())}) tidak cocok K_FOLDS={K_FOLDS}"

    synth_df = pd.read_csv(SYNTHETIC_MANIFEST_PATH)
    assert len(synth_df) == 504, f"Expected 504 baris sintetis, dapat {len(synth_df)} -- cek SYNTHETIC_MANIFEST_PATH"

    if SMOKE_TEST_N:
        print(f"\u26a0 MODE SMOKE TEST -- sintetis dipotong ke {SMOKE_TEST_N} baris, "
              f"train/test tiap fold juga dipotong. Set SMOKE_TEST_N = None utk full run.\n"
              f"\u26a0 INGAT: pakai OUTPUT_DIR terpisah dari full run.\n")
        synth_df = synth_df.iloc[:SMOKE_TEST_N].reset_index(drop=True)

    processor = Wav2Vec2Processor.from_pretrained(MMS_CHECKPOINT, target_lang=MMS_TARGET_LANG)
    check_vocab_coverage(processor, synth_df["text"].tolist(), label="sintetis (training)")
    check_vocab_coverage(processor, df["text"].tolist(), label="authentic (gabungan seluruh fold, train+test)")

    pred_out_path = os.path.join(OUTPUT_DIR, "combined_mms_predictions.csv")
    if os.path.exists(pred_out_path):
        print(f"Prediksi gabungan semua fold sudah ada -- SKIP total (resume): {pred_out_path}")
        return pd.read_csv(pred_out_path)

    fold_pred_paths = {
        fold: os.path.join(OUTPUT_DIR, f"predictions_combined_mms_fold{fold}.csv")
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
            fold_log_path = os.path.join(OUTPUT_DIR, f"log_history_combined_mms_fold{fold}.json")
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
        print(f"Test : {len(test_df)} segmen dari {test_df.recording_id.nunique()} rekaman")

        assert set(auth_train_df.recording_id) & set(test_df.recording_id) == set(), \
            f"BOCOR di fold {fold}: ada recording_id yang muncul di train DAN test!"

        effective_batch = PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
        total_train = len(auth_train_df) + len(synth_df)
        effective_epochs = MAX_STEPS * effective_batch / total_train
        print(f"Compute budget: {MAX_STEPS} step x {effective_batch} (effective batch) / "
              f"{total_train} contoh \u2248 {effective_epochs:.1f} epoch efektif")

        auth_ds = build_authentic_dataset(auth_train_df, processor)
        train_ds = concatenate_datasets([synth_ds, auth_ds]).shuffle(seed=RANDOM_SEED)

        model = Wav2Vec2ForCTC.from_pretrained(
            MMS_CHECKPOINT,
            target_lang=MMS_TARGET_LANG,
            ignore_mismatched_sizes=True,
            attention_dropout=0.0,
            hidden_dropout=0.0,
            feat_proj_dropout=0.0,
            layerdrop=0.0,
            ctc_loss_reduction="mean",
        )
        model.load_adapter(MMS_TARGET_LANG)
        verify_model_config(model, processor)
        freeze_for_adapter_only(model)
        model.to(device)

        data_collator = DataCollatorCTCWithPadding(processor=processor)

        fold_dir = os.path.join(OUTPUT_DIR, f"mms_fold{fold}")
        training_args_kwargs = dict(
            output_dir=fold_dir,
            per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            learning_rate=LEARNING_RATE,
            warmup_steps=WARMUP_STEPS,
            max_steps=MAX_STEPS,
            gradient_checkpointing=True,
            fp16=True,
            logging_steps=LOGGING_STEPS,
            save_strategy="no",     # model DIBUANG tiap fold -- SAMA PERSIS Authentic-Only/Whisper Combined
            eval_strategy="no",
            report_to=[],
            seed=RANDOM_SEED,
            remove_unused_columns=False,
        )
        if USE_GROUP_BY_LENGTH:
            ta_fields = {f.name for f in dataclasses.fields(TrainingArguments)}
            if "train_sampling_strategy" in ta_fields:
                training_args_kwargs["train_sampling_strategy"] = "group_by_length"
                training_args_kwargs["length_column_name"] = "input_length"
            elif "group_by_length" in ta_fields:
                training_args_kwargs["group_by_length"] = True
                training_args_kwargs["length_column_name"] = "input_length"
        training_args = TrainingArguments(**training_args_kwargs)

        trainer_kwargs = dict(
            model=model, args=training_args, train_dataset=train_ds, data_collator=data_collator,
        )
        sig_params = inspect.signature(Trainer.__init__).parameters
        if "processing_class" in sig_params:
            trainer_kwargs["processing_class"] = processor.feature_extractor
        elif "tokenizer" in sig_params:
            trainer_kwargs["tokenizer"] = processor.feature_extractor

        trainer = Trainer(**trainer_kwargs)
        trainer.train()
        fold_log_history = trainer.state.log_history
        all_log_histories[f"fold{fold}"] = fold_log_history

        fold_predictions = predict_fold(model, processor, test_df, fold, device)
        all_predictions.extend(fold_predictions)

        pd.DataFrame(fold_predictions).to_csv(fold_pred_path, index=False)
        fold_log_path = os.path.join(OUTPUT_DIR, f"log_history_combined_mms_fold{fold}.json")
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
        print(f"\n\u26a0 PERINGATAN: cuma {len(pred_df)}/106 segmen berhasil diprediksi. Cek sebelum lanjut evaluasi.")

    pred_df.to_csv(pred_out_path, index=False)
    print(f"Tersimpan: {pred_out_path}")

    log_out_path = os.path.join(OUTPUT_DIR, "train_log_history_all_folds_combined_mms.json")
    with open(log_out_path, "w") as f:
        json.dump(all_log_histories, f, indent=2)
    print(f"Tersimpan: {log_out_path}")

    return pred_df


if __name__ == "__main__":
    main()
