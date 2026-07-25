"""
finetune_whisper_authentic_colab.py

Tujuan
------
Fine-tuning Whisper-small ke kondisi Authentic-Only (106 segmen authentic),
dievaluasi lewat K-fold CV group-aware per rekaman (fold ditentukan oleh
kfold_split_authentic_colab.py, BUKAN dihitung ulang di sini -- satu sumber
kebenaran untuk semua model, lihat authentic_manifest_with_fold.csv).

Untuk tiap fold: model Whisper-small di-load FRESH dari pretrained (bukan
lanjut dari fold sebelumnya), di-fine-tune di (K-1) fold sisanya, lalu
dipakai generate prediksi mentah untuk fold yang disisihkan. Setelah semua
fold selesai, gabungan prediksi = 106 baris (tiap segmen diprediksi TEPAT 1x,
oleh model yang TIDAK PERNAH melihat rekaman asal segmen itu saat training).

Skrip ini CUMA urus training + generate teks mentah. WER/CER/bootstrap CI
dihitung terpisah oleh compute_wer_cer.py + bootstrap_ci.py yang sudah ada
(reuse langsung dari Zero-Shot, sudah group-aware) -- lihat catatan skema
kolom output di bawah.

============================================================================
KEPUTUSAN DESAIN & HYPERPARAMETER (dibahas dari nol sesi ini, baca ini dulu)
============================================================================

Konteks compute: Colab free tier, GPU T4 (~16GB VRAM). Single-seed per fold
(bukan multi-seed) -- keputusan eksplisit sesi ini, konsisten dengan Zero-Shot,
hemat compute untuk kondisi "smoke test" pipeline ini.

1) Compute-matched steps (MAX_STEPS = 600, effective batch = 16)
   Ini BUKAN cuma angka untuk Authentic-Only -- ini budget compute yang WAJIB
   dipakai ulang persis sama untuk Whisper di kondisi Synthetic-Only dan
   Combined nanti (lihat Peta Variabel 5: "compute budget (steps)" adalah
   variabel KONTROL, bukan variabel yang boleh beda per kondisi). Kalau nanti
   MAX_STEPS direvisi, revisi di KETIGA kondisi Whisper, bukan cuma satu.

   Implikasi effective epoch SANGAT beda antar kondisi (ini EXPECTED, bukan
   bug -- justru bagian dari apa yang mau diuji tesis: apakah lebih banyak
   data data dengan compute sama menghasilkan generalisasi lebih baik):
   - Authentic-Only (~85 segmen train/fold): 600 step x 16 / 85 ~ 113 epoch
     efektif -- TINGGI, risiko overfit ke 85 contoh nyata. Diterima secara
     sadar karena korpus authentic memang sekecil itu (bagian dari premis
     "low-resource").
   - Synthetic-Only (504 segmen, saat itu nanti): ~19 epoch efektif.
   - Combined (~589 segmen/fold nanti): ~16 epoch efektif.
   Catatan jujur untuk Limitasi tesis: skrip ini TIDAK pakai early stopping
   atau checkpoint selection berbasis validation (supaya compute-matched
   antar kondisi tetap ketat/adil) -- konsekuensinya, checkpoint akhir bisa
   saja bukan checkpoint "terbaik" kalau ada overfitting di step-step akhir.
   Log training loss per-fold disimpan (train_log_history_all_folds.json)
   supaya bisa diperiksa manual pasca-hoc, tapi TIDAK dipakai untuk memilih
   step/checkpoint secara adaptif.

2) Learning rate = 1e-5, warmup = 60 step
   Nilai standar dari resep fine-tuning Whisper full (bukan LoRA/adapter) di
   literatur/tutorial HF -- aman untuk whisper-small, tidak butuh diubah
   untuk jumlah step yang jauh lebih kecil karena baseline korpus fine-tune
   yang jadi rujukan (4-5k step) juga per-STEP magnitude-nya sama; yang beda
   di sini cuma total step, bukan LR per-step. Kalau train loss di
   train_log_history_all_folds.json terlihat nyaris tidak turun sama sekali
   di 600 step, pertimbangkan naikkan ke 3e-5 pada sesi revisi.

3) Batch size: per_device=8, grad_accum=2 (effective=16), fp16 ON,
   gradient_checkpointing ON -- dipilih konservatif untuk T4 16GB free tier
   (bukan Colab Pro A100), supaya tidak OOM meski RAM GPU lagi dipakai
   proses lain / kurang stabil di free tier. Turunkan ke per_device=4,
   grad_accum=4 kalau tetap OOM.

4) Bahasa/task decoding: language="id", task="transcribe" -- DICOCOKKAN
   PERSIS ke zero_shot_eval_whisper_mms_colab.py (sudah diverifikasi baca
   source-nya langsung, bukan tebakan lagi):
   - transcribe_whisper() di sana pakai default param language="id" (kode
     ISO pendek, bukan "indonesian" -- keduanya resolve ke token bahasa yang
     sama di WhisperTokenizer, tapi dipakai "id" di sini juga supaya
     pemanggilan API-nya identik persis, menghindari risiko sekecil apa pun).
   - model.generate() di sana dipanggil TANPA max_length dan TANPA num_beams
     eksplisit (cuma forced_decoder_ids) -- artinya pakai default
     generation_config bawaan checkpoint openai/whisper-small. Skrip ini
     MENIRU PERSIS: predict_fold() tidak lagi set max_length/num_beams
     sendiri, supaya strategi decoding Authentic-Only 100% sama dengan
     Zero-Shot (kalau ada gap WER, itu murni efek fine-tuning, bukan
     confound strategi decoding yang berbeda).
   - WhisperProcessor.from_pretrained() di zero-shot dipanggil TANPA
     language=/task= (di-set belakangan lewat get_decoder_prompt_ids saja) --
     skrip ini juga disesuaikan begitu.
   - Prediksi juga dilakukan PER-FILE (bukan batched) dengan try/except per
     item, meniru pola evaluate_model() di zero-shot -- kalau 1 file gagal
     (mis. audio corrupt), tidak menggagalkan seluruh fold, cuma di-skip
     dengan warning (lihat catatan skip di predict_fold()).

5) SKEMA KOLOM OUTPUT PREDIKSI -- SUDAH DIVERIFIKASI, BUKAN TEBAKAN LAGI
   compute_wer_cer.py (evaluate_manifest) dan bootstrap_ci.py
   (load_predictions_csv) SAMA-SAMA baca CSV dengan kolom file_id, reference,
   hypothesis lewat csv.DictReader -- kolom ekstra (recording_id, fold) di
   output skrip ini otomatis diabaikan, TIDAK bentrok. Sudah dites langsung:
   compute_wer_cer.py dijalankan apa adanya terhadap CSV berskema persis
   output skrip ini, hasilnya benar. Teks reference & hypothesis juga
   dinormalisasi (NFC + rapikan spasi) sebelum disimpan, meniru persis
   fungsi normalize() di zero_shot_eval_whisper_mms_colab.py -- supaya
   format artefak CSV Authentic-Only konsisten dengan Zero-Shot.

   CATATAN untuk evaluasi nanti (lihat evaluate_authentic_only_whisper.py):
   __main__ bootstrap_ci.py hardcode pola nama file Zero-Shot
   ("zeroshot_{model}_predictions.csv" di dalam RESULTS_DIR) -- TIDAK akan
   otomatis nemu "authentic_only_whisper_predictions.csv". Driver terpisah
   sudah dibuat supaya fungsi-fungsi di bootstrap_ci.py (bootstrap_ci(),
   load_predictions_csv(), load_recording_mapping(), resolve_groups())
   dipakai ulang APA ADANYA lewat import, bukan disalin ulang.

6) RESUME-CAPABLE (ditambahkan setelah run pertama sempat lama & khawatir
   Colab disconnect di tengah jalan): tiap fold selesai, hasilnya LANGSUNG
   ditulis ke disk (predictions_fold{N}.csv + log_history_fold{N}.json),
   BUKAN nunggu 5 fold kelar semua. Kalau proses berhenti di tengah (disconnect,
   crash, dsb), tinggal jalankan ulang CELL yang sama -- fold yang sudah
   selesai otomatis di-skip (dimuat dari disk, instan), lanjut training dari
   fold yang belum. Pola sama seperti generate_tts_colab.py di pipeline kamu.
   Kalau mau training ULANG dari nol (bukan resume), hapus dulu
   predictions_fold*.csv & log_history_fold*.json di OUTPUT_DIR.

Prasyarat sebelum run di Colab
-------------------------------
- Google Drive sudah di-mount (`from google.colab import drive; drive.mount('/content/drive')`)
  SEBELUM cell ini dijalankan -- semua path CONFIG di bawah nunjuk ke Drive
  (/content/drive/MyDrive/bugis_authentic/...), bukan /content/... lokal,
  supaya tahan disconnect (lihat catatan resume di poin 6).
- authentic_manifest_with_fold.csv (hasil kfold_split_authentic_colab.py) sudah
  diupload ke folder Drive itu juga.
- 106 file audio WAV (mono 16kHz 16-bit PCM) sudah diupload/di-mount, path-nya
  cocok dengan AUDIO_DIR + audio_filename di manifest.
- pip install: transformers, datasets, accelerate, librosa, soundfile
  (torch + GPU driver sudah tersedia default di runtime Colab GPU).

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
MANIFEST_WITH_FOLD_PATH = "/content/drive/MyDrive/bugis_authentic/authentic_manifest_with_fold.csv"   # dari kfold_split_authentic_colab.py -- taruh di Drive juga, bukan cuma di /content lokal
AUDIO_DIR = "/content/drive/MyDrive/bugis_authentic/audio"     # SAMA dengan AUDIO_DIR di zero_shot_eval_whisper_mms_colab.py
OUTPUT_DIR = "/content/drive/MyDrive/bugis_authentic/authentic_only_results"   # WAJIB di Drive (/content/drive/...), BUKAN /content/... lokal -- /content/... hilang total kalau runtime disconnect/reset, resume di bawah cuma berguna kalau hasilnya kesimpen di Drive

MODEL_NAME = "openai/whisper-small"
LANGUAGE = "id"              # SAMA PERSIS dengan default param di transcribe_whisper() zero-shot (bukan "indonesian")
TASK = "transcribe"

K_FOLDS = 5                  # HARUS sama dengan kfold_split_authentic_colab.py
RANDOM_SEED = 42             # single-seed (keputusan sesi ini)

# ---- Compute budget: WAJIB dipakai ulang identik utk Synthetic-Only & Combined (Whisper) ----
MAX_STEPS = 600
LEARNING_RATE = 1e-5
WARMUP_STEPS = 60
PER_DEVICE_TRAIN_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 2     # effective batch = 16
LOGGING_STEPS = 20

FP16 = True
GRADIENT_CHECKPOINTING = True
# --------------------------

# CATATAN: TIDAK ada GENERATION_MAX_LENGTH / num_beams di sini SENGAJA --
# predict_fold() meniru model.generate() zero-shot yang juga tidak set itu,
# supaya decoding 100% sama persis (lihat poin 4 di docstring atas).


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Padding input_features (mel spectrogram) & labels (token ids) terpisah,
    label padding di-mask jadi -100 supaya tidak ikut dihitung loss."""
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # BOS akan ditambahkan otomatis saat training (append_bos di tokenizer),
        # kalau sudah ada di label (dari versi tokenizer lama) buang biar tidak dobel.
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def normalize(text: str) -> str:
    """SAMA PERSIS dengan normalize() di zero_shot_eval_whisper_mms_colab.py
    (NFC + strip + rapikan spasi) -- supaya artefak CSV Authentic-Only
    format teksnya konsisten dengan Zero-Shot."""
    text = unicodedata.normalize("NFC", text.strip())
    return " ".join(text.split())


def load_audio(audio_filename: str) -> np.ndarray:
    path = os.path.join(AUDIO_DIR, audio_filename)
    array, _sr = librosa.load(path, sr=16000, mono=True)
    return array


def build_hf_dataset(df: pd.DataFrame, processor: WhisperProcessor) -> Dataset:
    """Konversi dataframe (file_id, text, audio_filename, ...) jadi HF Dataset
    berisi input_features (mel spectrogram) + labels (token ids)."""
    ds = Dataset.from_pandas(df[["file_id", "text", "audio_filename"]].reset_index(drop=True))

    def _prepare(example):
        audio_array = load_audio(example["audio_filename"])
        example["input_features"] = processor.feature_extractor(
            audio_array, sampling_rate=16000
        ).input_features[0]
        example["labels"] = processor.tokenizer(example["text"]).input_ids
        return example

    ds = ds.map(_prepare, remove_columns=["text", "audio_filename"])
    return ds


def transcribe_whisper_one(model, processor, forced_decoder_ids, audio_path, device) -> str:
    """Generate 1 file -- signature & isi meniru persis transcribe_whisper()
    di zero_shot_eval_whisper_mms_colab.py (beda cuma forced_decoder_ids
    dihitung sekali di luar loop, bukan per panggilan, karena tidak berubah)."""
    audio, _ = librosa.load(str(audio_path), sr=16000)
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features
    if device == "cuda":
        input_features = input_features.to("cuda")
    with torch.no_grad():
        predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
    return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]


def predict_fold(model, processor, forced_decoder_ids, test_df: pd.DataFrame, fold: int, device) -> List[Dict]:
    """Generate prediksi mentah untuk semua baris test_df, PER-FILE (bukan
    batched) dengan try/except -- meniru pola evaluate_model() di zero-shot,
    supaya 1 file gagal tidak menggagalkan seluruh fold."""
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
            print(f"  ⚠ GAGAL file_id={row.file_id} (fold {fold}): {e}")
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
              "Pastikan runtime Colab diset ke GPU (Runtime > Change runtime type > T4 GPU).")

    df = pd.read_csv(MANIFEST_WITH_FOLD_PATH)
    assert len(df) == 106, f"Expected 106 baris, dapat {len(df)}"
    assert set(df.fold.unique()) == set(range(K_FOLDS)), \
        f"fold di manifest ({sorted(df.fold.unique())}) tidak cocok K_FOLDS={K_FOLDS}"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    processor = WhisperProcessor.from_pretrained(MODEL_NAME)  # meniru zero-shot: tanpa language=/task= saat load
    forced_decoder_ids = processor.get_decoder_prompt_ids(language=LANGUAGE, task=TASK)

    all_predictions: List[Dict] = []
    all_log_histories: Dict[str, Any] = {}

    for fold in range(K_FOLDS):
        fold_pred_path = os.path.join(OUTPUT_DIR, f"predictions_fold{fold}.csv")
        fold_log_path = os.path.join(OUTPUT_DIR, f"log_history_fold{fold}.json")

        # RESUME: kalau fold ini sudah pernah selesai (predictions_fold{N}.csv ada),
        # skip training-nya sama sekali -- tinggal muat hasil lama dari disk. Jadi
        # kalau Colab disconnect di tengah fold 3, tinggal jalankan ulang CELL yang
        # sama: fold 0-2 langsung ke-skip (instan), lanjut training dari fold 3.
        # Pola sama seperti generate_tts_colab.py di pipeline kamu (resume-capable).
        if os.path.exists(fold_pred_path):
            print(f"\n{'=' * 60}\nFOLD {fold + 1}/{K_FOLDS} -- SUDAH ADA, DI-SKIP (resume)\n{'=' * 60}")
            fold_pred_df = pd.read_csv(fold_pred_path)
            all_predictions.extend(fold_pred_df.to_dict("records"))
            if os.path.exists(fold_log_path):
                all_log_histories[f"fold{fold}"] = json.loads(Path(fold_log_path).read_text(encoding="utf-8"))
            print(f"Dimuat dari: {fold_pred_path} ({len(fold_pred_df)} baris)")
            continue

        print(f"\n{'=' * 60}\nFOLD {fold + 1}/{K_FOLDS}\n{'=' * 60}")
        train_df = df[df.fold != fold].reset_index(drop=True)
        test_df = df[df.fold == fold].reset_index(drop=True)
        print(f"Train: {len(train_df)} segmen dari {train_df.recording_id.nunique()} rekaman")
        print(f"Test : {len(test_df)} segmen dari {test_df.recording_id.nunique()} rekaman "
              f"(TIDAK overlap dengan rekaman train di atas -- group-aware)")

        # sanity check leakage per fold, bukan cuma dipercaya dari file input
        assert set(train_df.recording_id) & set(test_df.recording_id) == set(), \
            f"BOCOR di fold {fold}: ada recording_id yang muncul di train DAN test!"

        train_ds = build_hf_dataset(train_df, processor)

        model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)  # fresh tiap fold
        model.config.forced_decoder_ids = None
        model.config.suppress_tokens = []
        model.to(device)

        data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

        fold_dir = os.path.join(OUTPUT_DIR, f"fold{fold}")
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
            save_strategy="no",     # compute-matched: cuma pakai checkpoint akhir, hemat disk
            eval_strategy="no",     # fold test TIDAK disentuh sama sekali selama training
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
        # Seq2SeqTrainer (yang lama masih jalan sampai suatu versi, lalu dibuang
        # total -- makanya dicek dinamis via signature, bukan hardcode salah satu,
        # supaya script ini tidak gampang rusak lagi kalau Colab update transformers).
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

        # SIMPAN LANGSUNG per fold (bukan nunggu 5 fold kelar) -- ini yang bikin
        # resume di atas bisa jalan. Kalau proses mati di fold ini SETELAH baris
        # ini tereksekusi, fold ini tetap dianggap "selesai" pas resume nanti.
        pd.DataFrame(fold_predictions).to_csv(fold_pred_path, index=False)
        Path(fold_log_path).write_text(json.dumps(fold_log_history, indent=2), encoding="utf-8")
        print(f"Fold {fold}: {len(fold_predictions)} prediksi tersimpan ke {fold_pred_path}")

        # bersihkan GPU sebelum fold berikutnya (penting di runtime tunggal T4 free)
        del model, trainer
        gc.collect()
        torch.cuda.empty_cache()

    pred_df = pd.DataFrame(all_predictions).sort_values("file_id").reset_index(drop=True)

    # sanity check akhir: idealnya 106/106 segmen terprediksi TEPAT 1x. TIDAK
    # di-assert keras (crash) karena predict_fold() sengaja skip-on-failure
    # per file (meniru zero-shot) -- kalau sampai training 5 fold selesai tapi
    # dibikin crash cuma gara-gara 1 file audio corrupt, hasil training yang
    # sudah mahal (waktu GPU) ikut hilang percuma. Cukup PERINGATKAN keras.
    assert pred_df.file_id.nunique() == len(pred_df), "Ada file_id yang terprediksi LEBIH dari 1x -- cek logika fold!"
    if len(pred_df) == 106:
        print(f"\nSanity check akhir: LOLOS (106/106 segmen terprediksi tepat 1x lewat skema k-fold)")
    else:
        print(f"\n⚠ PERINGATAN: cuma {len(pred_df)}/106 segmen berhasil diprediksi "
              f"(sisanya di-skip krn error, lihat log GAGAL di atas). Cek sebelum lanjut ke evaluasi.")

    pred_out_path = os.path.join(OUTPUT_DIR, "authentic_only_whisper_predictions.csv")
    pred_df.to_csv(pred_out_path, index=False)
    print(f"Tersimpan: {pred_out_path}")

    log_out_path = os.path.join(OUTPUT_DIR, "train_log_history_all_folds.json")
    with open(log_out_path, "w") as f:
        json.dump(all_log_histories, f, indent=2)
    print(f"Tersimpan: {log_out_path} (diagnostik loss curve per fold, TIDAK dipakai utk pilih checkpoint)")

    return pred_df


if __name__ == "__main__":
    main()
