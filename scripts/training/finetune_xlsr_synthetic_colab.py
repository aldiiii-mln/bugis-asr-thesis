"""
finetune_xlsr_synthetic_colab.py

STATUS: RESMI -- XLS-R-300m adalah MODEL KE-3, MENGGANTIKAN SeamlessM4T-medium
(keputusan dibuat setelah hasil lengkap Zero-Shot/Authentic-Only/Synthetic-Only/
Combined dibandingkan; SeamlessM4T dikeluarkan dari tesis). Dokumen metodologi
& rangkuman perlu diupdate mengikuti keputusan ini.

Tujuan
------
Fine-tuning XLS-R-300m Indonesia (Wikidepia/wav2vec2-xls-r-300m-indonesian)
ke kondisi Synthetic-Only: training memakai SEMUA 504 file sintetis
sekaligus (TANPA k-fold), lalu evaluasi LANGSUNG ke SEMUA 106 segmen
authentic. Pola strukturalnya identik dengan finetune_whisper_synthetic_colab.py
(single run, model disimpan) DIGABUNG dengan gaya CTC finetune_mms_synthetic_colab.py
(input_values, DataCollatorCTCWithPadding) -- TAPI strategi fine-tuning-nya
FULL FINE-TUNE spt Whisper (bukan adapter-only spt MMS), disalin PERSIS dari
finetune_xlsr_authentic_colab.py.

============================================================================
PERBEDAAN PENTING vs finetune_mms_synthetic_colab.py (baca ini)
============================================================================
- FULL FINE-TUNE, bukan adapter-only: dropout TIDAK dinolkan (beda dari MMS
  yang nolkan attention/hidden/feat_proj/layerdrop demi stabilitas adapter-
  only dgn parameter trainable sangat sedikit). Di sini transformer+CTC head
  penuh trainable, dropout default checkpoint tetap dipakai.
- `model.freeze_feature_encoder()` SAJA (bekukan CNN feature extractor doang)
  -- BUKAN `freeze_base_model()` (itu punya MMS, bekukan SELURUH base model,
  cuma cocok utk adapter-only).
- Checkpoint SUDAH fine-tuned ke Indonesia (bukan self-supervised mentah) --
  vocab/tokenizer checkpoint yang ada TETAP DIPAKAI, tidak dibangun ulang.
- Model DISIMPAN di akhir (final_model, spt Whisper Synthetic-Only) -- krn
  full fine-tune (~300M param, mirip ukuran Whisper-small 244M), BUKAN
  delta adapter kecil (beda dari SeamlessM4T/LoRA) ATAUPUN checkpoint besar
  969M (beda dari MMS) -- estimasi ukuran mirip kelas Whisper.

============================================================================
KONSISTENSI HYPERPARAMETER (WAJIB SAMA PERSIS dgn Authentic-Only XLS-R)
============================================================================
MAX_STEPS=150 [ESTIMASI, sama angka dgn MMS/SeamlessM4T demi kesederhanaan
cerita, TAPI keputusan independen -- lihat catatan panjang di
finetune_xlsr_authentic_colab.py soal kenapa num_train_epochs=30 blog resmi
TIDAK LANGSUNG SEBANDING], learning_rate=1e-4 [RESMI dari blog, tapi blog
itu utk checkpoint self-supervised mentah -- kita mulai dari checkpoint yg
SUDAH fine-tuned, jadi WAJIB smoke test dulu, jangan asumsi otomatis stabil],
weight_decay=0.005 [RESMI], warmup_steps=20 [ESTIMASI], per_device_batch=8,
grad_accum=4 (effective 32), gradient_checkpointing=True, fp16=True --
disalin PERSIS (variabel kontrol WAJIB sama across Authentic-Only/
Synthetic-Only/Combined, per model, lihat komentar
"WAJIB dipakai ulang identik utk Synthetic-Only & Combined (XLS-R)" di
finetune_xlsr_authentic_colab.py).

Implikasi epoch efektif: 150 step x 32 (effective batch) / 504 (file
sintetis) ~ 9.5 epoch efektif -- SAMA dengan MMS/SeamlessM4T Synthetic-Only
(compute budget & effective batch sama).

Decoding evaluasi: argmax langsung (TANPA beam search/LM) -- DICOCOKKAN
PERSIS ke zero_shot_eval_xlsr_colab.py & finetune_xlsr_authentic_colab.py.

============================================================================
CATATAN -- KOLOM AUDIO & PATH (SAMA dgn Whisper/MMS/SeamlessM4T Synthetic-Only)
============================================================================
synthetic_manifest_cleaned.csv tidak punya kolom audio_filename -- nama file
dibentuk dari file_id + ".mp3" (DIKONFIRMASI user: folder bugis_tts/audio,
flat, ekstensi .mp3).

BELUM PERNAH DITES END-TO-END dgn model asli -- sama status verifikasi dgn
finetune_xlsr_authentic_colab.py (saya tidak punya akses download checkpoint
dari huggingface.co di sandbox saya). Mekanika API (freeze_feature_encoder,
akses langsung .pad(), version-robust group_by_length) sudah diverifikasi
ke source code transformers oleh sesi sebelumnya yang bikin skrip
Authentic-Only-nya, dipakai ulang apa adanya di sini.

Prasyarat sebelum run di Colab
-------------------------------
- Google Drive sudah di-mount.
- synthetic_manifest_cleaned.csv, authentic_manifest.csv ada di path CONFIG.
- 504 file audio sintetis + 106 file audio authentic sudah ter-mount.
- pip install: transformers torch librosa soundfile accelerate jiwer -q
  (TIDAK butuh peft/loralib -- full fine-tune)
- SMOKE TEST WAJIB dulu (SMOKE_TEST_N) sebelum percaya full run.

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

from datasets import Dataset
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    TrainingArguments,
    Trainer,
    set_seed,
)

# ---- EDIT BAGIAN INI ----
SYNTHETIC_MANIFEST_PATH = "/content/drive/MyDrive/bugis_authentic/synthetic_manifest_cleaned.csv"   # cek lokasi persis
SYNTHETIC_AUDIO_DIR = "/content/drive/MyDrive/bugis_tts/audio"   # DIKONFIRMASI user
SYNTHETIC_AUDIO_EXT = ".mp3"    # DIKONFIRMASI user

AUTHENTIC_MANIFEST_PATH = "/content/drive/MyDrive/bugis_authentic/authentic_manifest.csv"   # test set TIDAK berubah
AUTHENTIC_AUDIO_DIR = "/content/drive/MyDrive/bugis_authentic/audio"

OUTPUT_DIR = "/content/drive/MyDrive/bugis_authentic/xlsr_synthetic_only_results"   # dedicated folder, konsisten pola xlsr_authentic_only_results
FINAL_MODEL_DIR = os.path.join(OUTPUT_DIR, "final_model")

XLSR_MODEL_TAG = "Wikidepia/wav2vec2-xls-r-300m-indonesian"

RANDOM_SEED = 42

# ---- Compute budget: SAMA PERSIS dgn finetune_xlsr_authentic_colab.py (variabel kontrol) ----
MAX_STEPS = 150
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.005
WARMUP_STEPS = 20
PER_DEVICE_TRAIN_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 4     # effective batch = 32
GRADIENT_CHECKPOINTING = True
FP16 = True

LOGGING_STEPS = 10
SAVE_STEPS = 50                     # checkpoint periodik utk resume (BARU dibanding Authentic-Only, spt Whisper Synthetic-Only)
SAVE_TOTAL_LIMIT = 2                # full fine-tune ~300M, mirip kelas Whisper-small -- SAVE_TOTAL_LIMIT longgar spt Whisper (BUKAN 1 spt MMS)

SUSPECT_CHARS_TO_CHECK = ["í"]      # dari pengalaman MMS/XLS-R Authentic-Only

SMOKE_TEST_N = None          # set angka kecil (mis. 3) utk smoke test dulu -- WAJIB, lihat docstring
# --------------------------


def normalize(text: str) -> str:
    """SAMA PERSIS dgn finetune_xlsr_authentic_colab.py."""
    text = unicodedata.normalize("NFC", text.strip())
    return " ".join(text.split())


def load_audio(path: str) -> np.ndarray:
    array, _sr = librosa.load(path, sr=16000, mono=True)
    return array


def synthetic_audio_path(file_id: str) -> str:
    """SAMA PERSIS dgn helper di skrip Synthetic-Only lain."""
    return os.path.join(SYNTHETIC_AUDIO_DIR, file_id + SYNTHETIC_AUDIO_EXT)


def check_vocab_coverage(texts: List[str], processor, label: str) -> None:
    """SAMA logikanya dgn check_vocab_coverage() di finetune_xlsr_authentic_colab.py,
    TAPI digeneralisasi terima list teks + label -- dipanggil 2x di main()
    (sintetis & authentic terpisah), SAMA PERSIS alasan dgn MMS Synthetic-Only:
    training text (sintetis) != test text (authentic), dua himpunan beda."""
    print(f"\n--- Cek cakupan vocabulary -- {label} (WAJIB dibaca) ---")
    all_text = " ".join(normalize(t) for t in texts)
    all_chars = set(all_text)
    vocab = processor.tokenizer.get_vocab()
    vocab_chars = {token for token in vocab.keys() if len(token) == 1}
    missing = sorted(c for c in all_chars if c != " " and c not in vocab_chars)
    print(f"  Total karakter unik di teks {label}: {len(all_chars)}")
    print(f"  Karakter TIDAK ada di vocab checkpoint: {missing if missing else '(tidak ada -- semua tercakup)'}")
    if missing:
        print(f"  \u26a0 PERINGATAN: {len(missing)} karakter akan dipetakan ke [UNK] -- model TIDAK BISA "
              f"belajar memprediksi karakter itu dgn benar. Cek dulu apakah ini penting sebelum lanjut.")
    for sc in SUSPECT_CHARS_TO_CHECK:
        status = "HILANG dari vocab" if sc in missing else ("ADA di vocab" if sc in vocab_chars else f"tidak muncul di teks {label}")
        print(f"  Karakter mencurigakan '{sc}': {status}")
    print("---\n")


@dataclass
class DataCollatorCTCWithPadding:
    """SAMA PERSIS dgn finetune_xlsr_authentic_colab.py."""
    processor: Any
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        batch = self.processor.feature_extractor.pad(input_features, padding=self.padding, return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad(label_features, padding=self.padding, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch


def build_synthetic_dataset(df: pd.DataFrame, processor) -> Dataset:
    """Konversi manifest sintetis (file_id, text) jadi HF Dataset. file_id
    dipaksa string -- FIX yang sama dgn skrip Combined (defensif, meski
    Synthetic-Only tidak concatenate_datasets, tetap konsisten)."""
    df = df.copy()
    df["file_id"] = df["file_id"].astype(str)
    ds = Dataset.from_pandas(df[["file_id", "text"]].reset_index(drop=True))

    def _prepare(example):
        audio_array = load_audio(synthetic_audio_path(example["file_id"]))
        example["input_values"] = processor.feature_extractor(
            audio_array, sampling_rate=16000
        ).input_values[0]
        example["labels"] = processor.tokenizer(normalize(example["text"])).input_ids
        return example

    ds = ds.map(_prepare, remove_columns=["text"])
    return ds


def transcribe_one(model, processor, audio_path, device) -> str:
    """SAMA PERSIS dgn finetune_xlsr_authentic_colab.py / zero-shot -- argmax
    langsung, TANPA beam search/LM."""
    audio, _ = librosa.load(str(audio_path), sr=16000)
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    input_values = inputs.input_values
    if device == "cuda":
        input_values = input_values.to("cuda")
    with torch.no_grad():
        logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    return processor.batch_decode(predicted_ids)[0]


def predict_all(model, processor, test_df: pd.DataFrame, device) -> List[Dict]:
    """Pola sama dgn predict_all() versi Whisper/MMS/SeamlessM4T Synthetic-Only."""
    model.eval()
    results = []
    rows = test_df.reset_index(drop=True)

    for row in rows.itertuples():
        audio_path = os.path.join(AUTHENTIC_AUDIO_DIR, row.audio_filename)
        reference = normalize(row.text)
        try:
            hypothesis = normalize(transcribe_one(model, processor, audio_path, device))
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
        print("PERINGATAN: GPU tidak terdeteksi -- XLS-R-300m di CPU akan sangat lambat.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    synth_df = pd.read_csv(SYNTHETIC_MANIFEST_PATH)
    assert len(synth_df) == 504, f"Expected 504 baris sintetis, dapat {len(synth_df)} -- cek SYNTHETIC_MANIFEST_PATH"

    auth_df = pd.read_csv(AUTHENTIC_MANIFEST_PATH)
    assert len(auth_df) == 106, f"Expected 106 baris authentic, dapat {len(auth_df)} -- cek AUTHENTIC_MANIFEST_PATH"

    if SMOKE_TEST_N:
        print(f"\u26a0 MODE SMOKE TEST -- training cuma {SMOKE_TEST_N} file sintetis, "
              f"eval cuma {SMOKE_TEST_N} file authentic. Set SMOKE_TEST_N = None utk full run.\n"
              f"\u26a0 SANGAT DISARANKAN (belum pernah dites end-to-end model asli). "
              f"INGAT pakai OUTPUT_DIR terpisah dari full run.\n")
        synth_df = synth_df.iloc[:SMOKE_TEST_N].reset_index(drop=True)
        auth_df = auth_df.iloc[:SMOKE_TEST_N].reset_index(drop=True)

    effective_batch = PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
    effective_epochs = MAX_STEPS * effective_batch / len(synth_df)
    print(f"Compute budget: {MAX_STEPS} step x {effective_batch} (effective batch) / "
          f"{len(synth_df)} file sintetis \u2248 {effective_epochs:.1f} epoch efektif")

    processor = Wav2Vec2Processor.from_pretrained(XLSR_MODEL_TAG)
    check_vocab_coverage(synth_df["text"].tolist(), processor, label="sintetis (training)")
    check_vocab_coverage(auth_df["text"].tolist(), processor, label="authentic (evaluasi)")

    pred_out_path = os.path.join(OUTPUT_DIR, "synthetic_only_xlsr_predictions.csv")

    if os.path.exists(pred_out_path) and os.path.exists(FINAL_MODEL_DIR):
        print(f"Model final & prediksi sudah ada -- SKIP total (resume).\n"
              f"  Model   : {FINAL_MODEL_DIR}\n  Prediksi: {pred_out_path}")
        return pd.read_csv(pred_out_path)

    if os.path.exists(FINAL_MODEL_DIR):
        print(f"Model final sudah tersimpan di {FINAL_MODEL_DIR} -- muat model itu, skip training.")
        model = Wav2Vec2ForCTC.from_pretrained(FINAL_MODEL_DIR)
        model.to(device)
    else:
        print(f"\n{'=' * 60}\nTRAINING -- {len(synth_df)} file sintetis, TANPA k-fold\n{'=' * 60}")
        print("Membangun dataset training...")
        train_ds = build_synthetic_dataset(synth_df, processor)

        model = Wav2Vec2ForCTC.from_pretrained(
            XLSR_MODEL_TAG,
            ctc_loss_reduction="mean",   # [RESMI] blog resmi
            pad_token_id=processor.tokenizer.pad_token_id,
        )
        # [RESMI, API drift diperbaiki] freeze_feature_encoder() SAJA -- BUKAN
        # freeze_base_model() (itu punya MMS, utk adapter-only). Dropout TIDAK
        # dinolkan -- full fine-tune, beda filosofi dari MMS adapter-only.
        model.freeze_feature_encoder()
        model.to(device)

        n_total = sum(p.numel() for p in model.parameters())
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Full fine-tuning (feature encoder dibekukan): {n_trainable:,} / {n_total:,} "
              f"parameter trainable ({100 * n_trainable / n_total:.1f}%)")

        data_collator = DataCollatorCTCWithPadding(processor=processor)

        # group_by_length -- version-robust, SAMA PERSIS pola finetune_xlsr_authentic_colab.py
        ta_fields = {f.name for f in dataclasses.fields(TrainingArguments)}
        if "train_sampling_strategy" in ta_fields:
            group_by_length_kwargs = {"train_sampling_strategy": "group_by_length"}
        elif "group_by_length" in ta_fields:
            group_by_length_kwargs = {"group_by_length": True}
        else:
            print("  \u26a0 Tidak ketemu field group_by_length ATAUPUN train_sampling_strategy -- dilewati.")
            group_by_length_kwargs = {}

        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            **group_by_length_kwargs,
            per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            warmup_steps=WARMUP_STEPS,
            max_steps=MAX_STEPS,
            gradient_checkpointing=GRADIENT_CHECKPOINTING,
            fp16=FP16,
            logging_steps=LOGGING_STEPS,
            save_strategy="steps",     # checkpoint periodik -- BEDA dari Authentic-Only ("no"), di sini
            save_steps=SAVE_STEPS,     # training 1x panjang & modelnya mau disimpan, butuh titik resume
            save_total_limit=SAVE_TOTAL_LIMIT,
            eval_strategy="no",
            report_to=[],
            seed=RANDOM_SEED,
            remove_unused_columns=False,
        )

        trainer_kwargs = dict(
            model=model, args=training_args, train_dataset=train_ds, data_collator=data_collator,
        )
        sig_params = inspect.signature(Trainer.__init__).parameters
        if "processing_class" in sig_params:
            trainer_kwargs["processing_class"] = processor.feature_extractor
        elif "tokenizer" in sig_params:
            trainer_kwargs["tokenizer"] = processor.feature_extractor

        trainer = Trainer(**trainer_kwargs)

        existing_checkpoints = sorted(Path(OUTPUT_DIR).glob("checkpoint-*"))
        resume = bool(existing_checkpoints)
        if resume:
            print(f"Checkpoint lama ditemukan ({len(existing_checkpoints)}) -- resume training dari checkpoint terakhir.")
        trainer.train(resume_from_checkpoint=resume)

        log_history = trainer.state.log_history
        Path(os.path.join(OUTPUT_DIR, "train_log_history_xlsr.json")).write_text(
            json.dumps(log_history, indent=2), encoding="utf-8"
        )

        os.makedirs(FINAL_MODEL_DIR, exist_ok=True)
        model.save_pretrained(FINAL_MODEL_DIR)
        processor.save_pretrained(FINAL_MODEL_DIR)
        print(f"Model final tersimpan ke: {FINAL_MODEL_DIR}")

    if os.path.exists(pred_out_path):
        print(f"Prediksi sudah ada di {pred_out_path} -- skip eval (resume).")
        pred_df = pd.read_csv(pred_out_path)
    else:
        print(f"\n{'=' * 60}\nEVALUASI -- {len(auth_df)} segmen authentic\n{'=' * 60}")
        predictions = predict_all(model, processor, auth_df, device)
        pred_df = pd.DataFrame(predictions).sort_values("file_id").reset_index(drop=True)

        assert pred_df.file_id.nunique() == len(pred_df), "Ada file_id terprediksi LEBIH dari 1x!"
        if len(pred_df) == len(auth_df):
            print(f"\nSanity check akhir: LOLOS ({len(pred_df)}/{len(auth_df)} segmen terprediksi tepat 1x)")
        else:
            print(f"\n\u26a0 PERINGATAN: cuma {len(pred_df)}/{len(auth_df)} segmen berhasil diprediksi.")

        pred_df.to_csv(pred_out_path, index=False)
        print(f"Tersimpan: {pred_out_path}")

    del model
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return pred_df


if __name__ == "__main__":
    main()
