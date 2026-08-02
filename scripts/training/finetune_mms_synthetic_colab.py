"""
finetune_mms_synthetic_colab.py

Tujuan
------
Fine-tuning MMS (facebook/mms-1b-all, adapter 'ind', adapter-only) ke kondisi
Synthetic-Only: training memakai SEMUA 504 file sintetis sekaligus (TANPA
k-fold), lalu evaluasi LANGSUNG ke SEMUA 106 segmen authentic (test set yang
SAMA dengan Zero-Shot & Authentic-Only). Pola strukturalnya identik dengan
finetune_whisper_synthetic_colab.py -- baca docstring skrip itu dulu kalau
belum, penjelasan di sini fokus ke hal yang KHUSUS MMS saja.

============================================================================
PERBEDAAN STRUKTURAL vs finetune_mms_authentic_colab.py
============================================================================
- TIDAK ADA loop fold -- alasan sama persis dgn Whisper Synthetic-Only:
  training (sintetis) & test (authentic) sumbernya beda total, tidak ada
  risiko kebocoran.
- Cuma 1 model dilatih, DISIMPAN ke FINAL_MODEL_DIR (beda dari
  Authentic-Only yang modelnya dibuang tiap fold).
- Resume: checkpoint HF Trainer (save_strategy="steps"), skip training+eval
  total kalau FINAL_MODEL_DIR + predictions CSV sudah ada.
- CATATAN UKURAN CHECKPOINT: MMS 964M parameter jauh lebih besar dari
  Whisper-small (244M) -- checkpoint penuh (bukan cuma delta adapter, karena
  freeze_base_model()/_get_adapters() BUKAN PEFT, jadi save_pretrained()
  tetap menulis SEMUA parameter termasuk yang dibekukan) makan tempat lebih
  banyak di Drive. SAVE_TOTAL_LIMIT diset 1 (bukan 2 seperti Whisper) demi
  hemat kuota, dan SAVE_STEPS lebih rapat (50, bukan 100) karena MAX_STEPS
  MMS jauh lebih kecil (150, bukan 600) -- tetap dapat ~3 titik checkpoint.

============================================================================
KONSISTENSI HYPERPARAMETER (WAJIB SAMA PERSIS dgn Authentic-Only MMS)
============================================================================
MAX_STEPS=150, learning_rate=1e-3, warmup_steps=20, per_device_batch=4,
grad_accum=8 (effective 32), group_by_length=True (via toggle
USE_GROUP_BY_LENGTH, deteksi otomatis nama parameter train_sampling_strategy
vs group_by_length antar versi transformers -- SAMA PERSIS pola dgn
Authentic-Only), dropout semua 0, ctc_loss_reduction="mean",
load_adapter('ind') (bukan init acak) -- disalin PERSIS dari
finetune_mms_authentic_colab.py (variabel kontrol). JANGAN diubah sepihak.

Implikasi epoch efektif: 150 step x 32 (effective batch) / 504 (file
sintetis) ~ 9.5 epoch efektif -- jauh lebih rendah dari Authentic-Only
(~56 epoch efektif ke ~85 segmen train/fold). Dicetak di runtime.

Decoding evaluasi: greedy argmax (TANPA beam search) -- DICOCOKKAN PERSIS ke
zero_shot_eval_whisper_mms_colab.py & finetune_mms_authentic_colab.py.

============================================================================
CEK VOCABULARY -- DUA SISI (BEDA dari Authentic-Only, baca ini)
============================================================================
Authentic-Only cuma cek vocab 1x (training text == test text pool, sama-sama
authentic). Synthetic-Only training text (sintetis, Bugis-proxy via TTS
Indonesia) dan test text (authentic asli) adalah DUA himpunan teks yang beda
-- jadi skrip ini cek vocab coverage KE DUA-DUANYA secara terpisah (label
"sintetis (training)" dan "authentic (evaluasi)"), supaya kalau ada karakter
yang hilang, ketahuan dari sisi mana asalnya. Vocab MMS ('ind') sendiri
TIDAK berubah oleh adapter fine-tuning (bukan train karakter baru ke
tokenizer), jadi baik training maupun evaluasi sama-sama kena "lantai error"
yang sama dari karakter yang hilang -- konsisten dgn temuan Authentic-Only
(karakter 'í' hilang dari vocab 'ind', lihat rangkuman metodologi 10.3).

============================================================================
CATATAN -- KOLOM AUDIO & PATH (SAMA dgn Whisper Synthetic-Only, sudah
dikonfirmasi user)
============================================================================
synthetic_manifest_cleaned.csv tidak punya kolom audio_filename -- nama file
dibentuk dari file_id + ".mp3" (DIKONFIRMASI user via screenshot Drive:
folder bugis_tts/audio, flat, ardi+gadis gabung, ekstensi .mp3).

Prasyarat sebelum run di Colab
-------------------------------
- Google Drive sudah di-mount.
- synthetic_manifest_cleaned.csv, authentic_manifest.csv ada di path CONFIG.
- 504 file audio sintetis di SYNTHETIC_AUDIO_DIR, 106 file audio authentic
  di AUTHENTIC_AUDIO_DIR.
- pip install: transformers datasets accelerate librosa soundfile
- Download pertama facebook/mms-1b-all lumayan besar -- sekali cache, run
  berikutnya (termasuk resume) baca dari cache lokal.

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
SYNTHETIC_MANIFEST_PATH = "/content/drive/MyDrive/bugis_authentic/synthetic_manifest_cleaned.csv"   # cek lokasi persis -- lihat catatan di respons chat
SYNTHETIC_AUDIO_DIR = "/content/drive/MyDrive/bugis_tts/audio"   # DIKONFIRMASI user: folder flat, ardi+gadis gabung
SYNTHETIC_AUDIO_EXT = ".mp3"    # DIKONFIRMASI user

AUTHENTIC_MANIFEST_PATH = "/content/drive/MyDrive/bugis_authentic/authentic_manifest.csv"   # test set TIDAK berubah
AUTHENTIC_AUDIO_DIR = "/content/drive/MyDrive/bugis_authentic/audio"    # SAMA dgn Whisper/zero-shot

OUTPUT_DIR = "/content/drive/MyDrive/bugis_authentic/synthetic_only_results"   # folder SAMA dgn Whisper Synthetic-Only, nama file beda jadi aman digabung
FINAL_MODEL_DIR = os.path.join(OUTPUT_DIR, "final_model_mms")   # DIPISAH dari final_model Whisper (nama beda)

MMS_CHECKPOINT = "facebook/mms-1b-all"
MMS_TARGET_LANG = "ind"      # konsisten Zero-Shot & Authentic-Only

RANDOM_SEED = 42

# ---- Compute budget: SAMA PERSIS dgn finetune_mms_authentic_colab.py (variabel kontrol) ----
MAX_STEPS = 150
LEARNING_RATE = 1e-3
WARMUP_STEPS = 20
PER_DEVICE_TRAIN_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 8     # effective batch = 32
USE_GROUP_BY_LENGTH = True

LOGGING_STEPS = 10

# ---- Checkpoint periodik (BARU dibanding Authentic-Only, lihat catatan
#      "UKURAN CHECKPOINT" di atas -- MMS lebih besar dari Whisper) ----
SAVE_STEPS = 50
SAVE_TOTAL_LIMIT = 1

FP16 = True
GRADIENT_CHECKPOINTING = True

SMOKE_TEST_N = None          # set angka kecil (mis. 3) utk smoke test dulu
# --------------------------


def normalize(text: str) -> str:
    """SAMA PERSIS dgn zero_shot_eval_whisper_mms_colab.py / finetune_mms_authentic_colab.py."""
    text = unicodedata.normalize("NFC", text.strip())
    return " ".join(text.split())


def load_audio(path: str) -> np.ndarray:
    array, _sr = librosa.load(path, sr=16000, mono=True)
    return array


def synthetic_audio_path(file_id: str) -> str:
    """SAMA PERSIS dgn helper di finetune_whisper_synthetic_colab.py."""
    return os.path.join(SYNTHETIC_AUDIO_DIR, file_id + SYNTHETIC_AUDIO_EXT)


def check_vocab_coverage(processor: Wav2Vec2Processor, texts: List[str], label: str) -> None:
    """SAMA logikanya dgn finetune_mms_authentic_colab.py, TAPI dipanggil 2x
    di main() (sintetis & authentic terpisah) -- lihat catatan "CEK
    VOCABULARY" di docstring atas kenapa dipisah."""
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
        print("  Karakter ini TIDAK PERNAH bisa muncul di prediksi MMS (zero-shot maupun fine-tuned) --")
        print("  WAJIB dicatat di Limitasi tesis kalau ini termasuk karakter bermakna (mis. apostrof glottal stop).")
    else:
        print(f"Semua karakter di teks {label} ADA di vocab -- aman.")
    print("---\n")


def verify_model_config(model, processor) -> bool:
    """SAMA PERSIS dgn finetune_mms_authentic_colab.py -- verifikasi
    eksplisit kombinasi kwargs (dropout/ctc_loss_reduction) beneran
    kepasang setelah from_pretrained(), bukan diasumsikan."""
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
        print("\u26a0\u26a0\u26a0 PERINGATAN KERAS: ada ketidakcocokan di atas -- JANGAN lanjut training, "
              "screenshot output ini & laporkan balik, jangan asumsikan tetap aman.")
    else:
        print("Semua cocok -- kombinasi kwargs beneran kepasang, aman lanjut.")
    print("---\n")
    return all_ok


def freeze_for_adapter_only(model: Wav2Vec2ForCTC) -> None:
    """SAMA PERSIS dgn finetune_mms_authentic_colab.py -- method resmi HF
    (freeze_base_model() + _get_adapters()), bukan reimplementasi manual."""
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
        print("\u26a0\u26a0\u26a0 PERINGATAN KERAS: 0 parameter trainable! model._get_adapters() kosong -- "
              "kemungkinan config.adapter_attn_dim tidak ter-set. JANGAN lanjut training, cek dulu manual.")


@dataclass
class DataCollatorCTCWithPadding:
    """SAMA PERSIS dgn finetune_mms_authentic_colab.py."""
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


def build_synthetic_dataset(df: pd.DataFrame, processor: Wav2Vec2Processor) -> Dataset:
    """Konversi manifest sintetis (file_id, text) jadi HF Dataset berisi
    input_values (raw waveform, bukan mel spectrogram -- CTC) + labels."""
    ds = Dataset.from_pandas(df[["file_id", "text"]].reset_index(drop=True))

    def _prepare(example):
        audio_array = load_audio(synthetic_audio_path(example["file_id"]))
        example["input_values"] = processor.feature_extractor(
            audio_array, sampling_rate=16000
        ).input_values[0]
        example["input_length"] = len(example["input_values"])   # dipakai group_by_length
        example["labels"] = processor.tokenizer(example["text"]).input_ids
        return example

    ds = ds.map(_prepare, remove_columns=["text"])
    return ds


def transcribe_mms_one(model, processor, audio_path, device) -> str:
    """SAMA PERSIS dgn finetune_mms_authentic_colab.py / zero-shot -- greedy
    argmax, TANPA beam search."""
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
    """Generate prediksi mentah utk SEMUA segmen authentic, PER-FILE dengan
    try/except -- pola sama dgn predict_all() versi Whisper Synthetic-Only."""
    model.eval()
    results = []
    rows = test_df.reset_index(drop=True)

    for row in rows.itertuples():
        audio_path = os.path.join(AUTHENTIC_AUDIO_DIR, row.audio_filename)
        reference = normalize(row.text)
        try:
            hypothesis = normalize(transcribe_mms_one(model, processor, audio_path, device))
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
        print("PERINGATAN: GPU tidak terdeteksi -- training di CPU akan SANGAT lambat, "
              "dan MMS-1b jauh lebih berat dari Whisper-small. Pastikan runtime GPU aktif.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    synth_df = pd.read_csv(SYNTHETIC_MANIFEST_PATH)
    assert len(synth_df) == 504, f"Expected 504 baris sintetis, dapat {len(synth_df)} -- cek SYNTHETIC_MANIFEST_PATH"

    auth_df = pd.read_csv(AUTHENTIC_MANIFEST_PATH)
    assert len(auth_df) == 106, f"Expected 106 baris authentic, dapat {len(auth_df)} -- cek AUTHENTIC_MANIFEST_PATH"

    if SMOKE_TEST_N:
        print(f"\u26a0 MODE SMOKE TEST -- training cuma {SMOKE_TEST_N} file sintetis, "
              f"eval cuma {SMOKE_TEST_N} file authentic. Set SMOKE_TEST_N = None utk full run.\n"
              f"\u26a0 INGAT: pakai OUTPUT_DIR terpisah dari full run (resume logic bisa salah kira selesai).\n")
        synth_df = synth_df.iloc[:SMOKE_TEST_N].reset_index(drop=True)
        auth_df = auth_df.iloc[:SMOKE_TEST_N].reset_index(drop=True)

    effective_batch = PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
    effective_epochs = MAX_STEPS * effective_batch / len(synth_df)
    print(f"Compute budget: {MAX_STEPS} step x {effective_batch} (effective batch) / "
          f"{len(synth_df)} file sintetis \u2248 {effective_epochs:.1f} epoch efektif")

    processor = Wav2Vec2Processor.from_pretrained(MMS_CHECKPOINT, target_lang=MMS_TARGET_LANG)
    check_vocab_coverage(processor, synth_df["text"].tolist(), label="sintetis (training)")
    check_vocab_coverage(processor, auth_df["text"].tolist(), label="authentic (evaluasi)")

    pred_out_path = os.path.join(OUTPUT_DIR, "synthetic_only_mms_predictions.csv")

    # ---- CEK APAKAH SUDAH SELESAI TOTAL (resume: skip training+eval sepenuhnya) ----
    if os.path.exists(pred_out_path) and os.path.exists(FINAL_MODEL_DIR):
        print(f"Model final & prediksi sudah ada -- SKIP total (resume).\n"
              f"  Model   : {FINAL_MODEL_DIR}\n  Prediksi: {pred_out_path}")
        return pd.read_csv(pred_out_path)

    # ---- TRAINING (dgn resume dari checkpoint kalau ada) ----
    if os.path.exists(FINAL_MODEL_DIR):
        print(f"Model final sudah tersimpan di {FINAL_MODEL_DIR} -- muat model itu, skip training.\n"
              f"CATATAN: model disimpan lewat save_pretrained() penuh (bukan PEFT), jadi reload apa\n"
              f"adanya via from_pretrained() -- TIDAK perlu load_adapter()/freeze ulang, arsitektur &\n"
              f"bobot adapter yg sudah dilatih ikut tersimpan di checkpoint. Ini asumsi standar\n"
              f"perilaku save_pretrained() HF, belum sempat dites end-to-end reload krn keterbatasan\n"
              f"akses model asli di sandbox saya -- kalau prediksi hasil reload ANEH/identik zero-shot,\n"
              f"screenshot & laporkan balik (pola bug yang sama persis pernah kejadian di OWSM).")
        model = Wav2Vec2ForCTC.from_pretrained(FINAL_MODEL_DIR)
        model.to(device)
        n_total = sum(p.numel() for p in model.parameters())
        print(f"Model dimuat: {n_total:,} parameter total.")
    else:
        print(f"\n{'=' * 60}\nTRAINING -- {len(synth_df)} file sintetis, TANPA k-fold\n{'=' * 60}")
        print("Membangun dataset training...")
        train_ds = build_synthetic_dataset(synth_df, processor)

        model = Wav2Vec2ForCTC.from_pretrained(
            MMS_CHECKPOINT,
            target_lang=MMS_TARGET_LANG,
            ignore_mismatched_sizes=True,
            attention_dropout=0.0,
            hidden_dropout=0.0,
            feat_proj_dropout=0.0,
            layerdrop=0.0,
            ctc_loss_reduction="mean",   # default library "sum" -- WAJIB dioverride
        )
        model.load_adapter(MMS_TARGET_LANG)   # lanjutkan dari adapter 'ind' yg sudah dipretrain, konsisten Zero-Shot
        verify_model_config(model, processor)   # WAJIB lolos sebelum lanjut
        freeze_for_adapter_only(model)
        model.to(device)

        data_collator = DataCollatorCTCWithPadding(processor=processor)

        training_args_kwargs = dict(
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
            eval_strategy="no",
            report_to=[],
            seed=RANDOM_SEED,
            remove_unused_columns=False,
        )
        # group_by_length: nama parameter berubah antar versi transformers --
        # dicek dinamis, SAMA PERSIS pola dgn finetune_mms_authentic_colab.py.
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
            model=model,
            args=training_args,
            train_dataset=train_ds,
            data_collator=data_collator,
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
        Path(os.path.join(OUTPUT_DIR, "train_log_history_mms.json")).write_text(
            json.dumps(log_history, indent=2), encoding="utf-8"
        )

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
        predictions = predict_all(model, processor, auth_df, device)
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
