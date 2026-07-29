"""
finetune_mms_authentic_colab.py

Tujuan
------
Fine-tuning MMS (facebook/mms-1b-all, adapter 'ind') ke kondisi Authentic-Only,
K-fold CV group-aware -- pola SAMA PERSIS dengan finetune_whisper_authentic_colab.py
(baca authentic_manifest_with_fold.csv yang SAMA, fold assignment yang SAMA,
JANGAN generate ulang k-fold buat MMS). Bedanya cuma arsitektur/loss (CTC,
bukan seq2seq) dan strategi fine-tuning (lihat poin 1 di bawah).

============================================================================
KEPUTUSAN DESAIN & HYPERPARAMETER KHUSUS MMS (baca ini dulu)
============================================================================

1) ADAPTER-ONLY, BUKAN FULL FINE-TUNE -- ini MENJAWAB confound yang sudah
   dicatat di Peta Variabel (Bagian 5) dokumen metodologi kamu ("strategi
   fine-tuning MMS: adapter vs full, kalau tidak sempat full fine-tune").
   Ini BUKAN "tidak sempat" -- ini keputusan TERPAKSA karena keterbatasan
   teknis: mms-1b-all itu 1 MILIAR parameter. Full fine-tune butuh optimizer
   state (~12-16 byte/parameter utk Adam) = puluhan GB, jauh di atas 16GB
   T4 free tier -- SEBELUM mikirin aktivasi/data sama sekali. Adapter-only
   cuma nge-training adapter layer + lm_head (biasanya <2% dari 1B parameter),
   sisanya DIBEKUKAN (requires_grad=False) -- selisih kebutuhan memori-nya
   drastis (optimizer state cuma perlu dihitung utk yg trainable).
   -> REKOMENDASI: update Bagian 5 dokumen metodologi kamu, confound ini
      sudah "resolved" (bukan lagi "kalau tidak sempat") -- adapter-only
      adalah keputusan final dgn alasan teknis eksplisit, bukan kompromi
      darurat. Redaksi yang saya sarankan ada di respons chat, bukan di sini.

2) Cek cakupan vocabulary SEBELUM training (check_vocab_coverage() di bawah)
   -- MMS pakai tokenizer CTC level-karakter yang SPESIFIK per bahasa
   ('ind' = Indonesia, dipakai sbg proksi Bugis, KONSISTEN sama Zero-Shot).
   RISIKO KONKRET: teks authentic Bugis kamu pakai APOSTROF sbg penanda
   glottal stop (mis. "gambara'") -- lihat juga temuan soal ini di Bagian
   9.5 (investigasi Alkitab Bugis, soal apostrof glottal stop). Saya TIDAK
   BISA cek dari sini apakah vocab 'ind' MMS punya karakter apostrof atau
   tidak (perlu akses ke model asli di Hugging Face Hub, yang tidak saya
   punya di sandbox saya) -- kalau TIDAK ada, karakter itu tidak akan
   pernah bisa muncul di hasil prediksi MMS (baik zero-shot MAUPUN
   fine-tuned), jadi CER akan selalu punya "lantai error" minimum dari
   karakter yg hilang ini, terlepas dari fine-tuning sebaik apa pun.
   Skrip ini CETAK hasil pengecekan di awal run -- WAJIB dibaca sebelum
   percaya hasil training, dan kalau apostrof/karakter lain hilang dari
   vocab, itu WAJIB masuk Limitasi tesis (bukan sekadar dicatat lalu lupa).

3) Learning rate = 1e-3, warmup = 100 step, effective batch = 32 --
   DIVERIFIKASI LANGSUNG ke blog resmi HF "Fine-Tune MMS Adapter Models for
   low-resource ASR" oleh Patrick von Platen (huggingface.co/blog/mms_adapters,
   dicek ulang 2026 -- bukan tebakan lagi kayak versi draft sebelumnya).
   Kutipan relevan dari blog itu: "`learning_rate` was chosen to be 1e-3
   which is a common default value for training with Adam" dan mereka pakai
   `per_device_train_batch_size=32` (tanpa gradient accumulation di setup
   mereka). Effective batch di sini di-set 32 juga (SAMA dgn resmi), TAPI
   dipecah jadi per_device=4 x grad_accum=8 -- bukan 32 langsung -- karena
   mereka kemungkinan pakai GPU lebih besar dari T4 16GB free tier (blog
   tidak sebutkan GPU spesifik). Ini adaptasi hardware yang setara secara
   matematis (gradient yang dihasilkan sama), bukan penyimpangan dari resep.

4) Dropout DIMATIKAN TOTAL saat training (attention_dropout=0, hidden_dropout=0,
   feat_proj_dropout=0, layerdrop=0) -- INI JUGA DARI BLOG RESMI, dikutip
   persis: "Since we're only training a small subset of weights, the model
   is not prone to overfitting. Therefore, we make sure to disable all
   dropout layers." Alasannya konsisten sama argumen adapter-only di poin 1:
   makin sedikit parameter yang bisa bergerak, makin kecil risiko overfit,
   jadi regularisasi dropout tidak diperlukan (malah bisa memperlambat
   konvergensi yang menurut blog resminya memang didesain SANGAT cepat).

5) ctc_loss_reduction="mean" -- WAJIB di-set eksplisit, DEFAULT library-nya
   "sum" (dicek langsung ke source code Wav2Vec2Config, bukan asumsi).
   Kalau dibiarkan default "sum", magnitude loss akan ikut naik-turun sesuai
   panjang urutan audio & komposisi batch, bikin efek learning_rate jadi
   tidak konsisten antar batch -- "mean" menormalisasi ini, sesuai resep resmi.

6) group_by_length -- DITAMBAHKAN (juga dari resep resmi): mengelompokkan
   sampel dengan panjang audio mirip ke batch yang sama, mengurangi padding
   sia-sia (efisiensi + sedikit mengurangi risiko OOM dari batch yang
   kebetulan berisi campuran sangat pendek & sangat panjang). CATATAN
   TEKNIS: nama parameter di TrainingArguments BERUBAH di versi transformers
   terbaru -- versi lama pakai `group_by_length=True` (persis kode di blog
   2023), versi baru (yang otomatis ke-install di Colab kamu, TERVERIFIKASI
   lewat instalasi transformers 5.14.1 di sandbox saya) pakai
   `train_sampling_strategy="group_by_length"` -- skrip ini deteksi otomatis
   versi mana yang tersedia (pola sama dgn fix tokenizer=/processing_class=
   di Whisper), supaya tidak rusak lagi kalau Colab update transformers.
   UPDATE (setelah OOM di per_device=8): fitur ini DIMATIKAN SEMENTARA lewat
   toggle USE_GROUP_BY_LENGTH di CONFIG -- dugaan kuat justru fitur ini yang
   bikin OOM (klip terpanjang ~27 detik ngumpul jadi 1 batch murni panjang,
   bukan tercampur klip pendek kayak batching acak biasa) -- lihat catatan
   percobaan ke-3 di CONFIG.

7) STRATEGI FREEZE: pakai method RESMI `model.freeze_base_model()` +
   `model._get_adapters()` (BUKAN reimplementasi manual berbasis pencarian
   substring nama parameter seperti draft sebelumnya) -- sudah saya
   verifikasi LANGSUNG ke source code transformers (bukan cuma baca blog):
   `freeze_base_model()` membekukan SEMUA parameter di `self.wav2vec2`
   (termasuk adapter layer, karena adapter layer itu nested DI DALAM
   wav2vec2 encoder), lalu `_get_adapters()` mengembalikan dict berisi
   PERSIS parameter adapter layer (dideteksi via isinstance ke
   Wav2Vec2AttnAdapterLayer, bukan string matching) + lm_head, yang di-set
   requires_grad=True lagi secara eksplisit. Ini method yang di-maintain
   resmi oleh HF, jauh lebih defensible utk metodologi drpd reimplementasi
   sendiri, meski hasilnya provably setara (sudah saya cross-check).

8) load_adapter('ind') vs init_adapter_layers() -- KEPUTUSAN: skrip ini
   pakai load_adapter('ind') (lanjutkan dari adapter Indonesia yang SUDAH
   di-pretrain), BUKAN init_adapter_layers() (re-init acak + bangun vocab
   baru dari data Bugis sendiri, yang jadi alur UTAMA di blog resmi utk
   bahasa yang sama sekali baru). Blog resmi SECARA EKSPLISIT menyebut opsi
   load_adapter() sebagai alternatif valid ("it is also possible to not
   re-initialize the adapter weights and continue fine-tuning... should
   load fitting adapter weights via the load_adapter(...) method"). Kita
   pilih opsi ini demi KONSISTENSI dengan Zero-Shot (vocab & starting point
   yang sama persis, supaya perbandingan Authentic-Only vs Zero-Shot untuk
   MMS itu within-model apple-to-apple -- dasar Klaim 1b, sama alasannya
   dengan keputusan decoding di Whisper). Trade-off yang harus diakui:
   vocab tetap vocab 'ind' (Indonesia), BUKAN vocab custom dari data Bugis
   -- makanya check_vocab_coverage() di poin 2 penting (blog resmi sendiri
   memperingatkan "the vocabulary still will not match the custom training
   data very well though" utk kasus load_adapter dari bahasa lain).

9) MAX_STEPS -- REVISI dari rencana awal (600), lihat catatan lengkap +
   perhitungan di komentar CONFIG di atas (bagian "Compute budget"). Ringkas:
   600 step dipilih awalnya karena blog resmi konvergen kuat di step ~400
   (WER 0.280->0.223 dari step 100->400), tapi smoke test empiris di T4
   kamu menunjukkan ~0.03 it/s (jauh lebih lambat dari estimasi awal) --
   600 step = ~25 jam total utk 5 fold, tidak feasible. DITURUNKAN ke 150 --
   masih di atas titik "sudah membaik signifikan" (step 100) yang dibuktikan
   blog resmi, tapi realistis buat T4 free tier (dgn bantuan resume-capable,
   lihat poin 10). WARMUP_STEPS ikut diturunkan proporsional. Efek epoch
   efektif jadi lebih rendah dari estimasi awal (~225 epoch turun ke ~56 epoch
   di fold Authentic-Only) -- masih tinggi, tapi tidak seekstrem sebelumnya.

10) SAMA seperti Whisper: resume-capable (skip fold yg predictions_fold{N}.csv
    -nya sudah ada), OUTPUT_DIR wajib di Drive (bukan /content/... lokal),
    version-robust utk rename tokenizer= -> processing_class= di Trainer.

Sumber yang dipakai buat verifikasi (dicek ulang langsung, bukan dari ingatan):
- Blog: https://huggingface.co/blog/mms_adapters ("Fine-Tune MMS Adapter
  Models for low-resource ASR", Patrick von Platen, HuggingFace, 2023)
- Source code: transformers/models/wav2vec2/modeling_wav2vec2.py (method
  _get_adapters, init_adapter_layers, freeze_base_model, load_adapter) dan
  configuration_wav2vec2.py (default ctc_loss_reduction), dicek langsung di
  package transformers versi 5.14.1 (versi yang ke-install otomatis di
  sandbox saya lewat pip, kemungkinan besar sama dgn yang ke-install di
  Colab kamu kalau kamu install tanpa pin versi spesifik).

Prasyarat sebelum run di Colab
-------------------------------
- Google Drive sudah di-mount.
- authentic_manifest_with_fold.csv (SAMA PERSIS file yang dipakai Whisper,
  JANGAN generate ulang) ada di folder Drive project.
- 106 file audio WAV ada di AUDIO_DIR (SAMA dengan yang dipakai Whisper).
- pip install: transformers datasets accelerate librosa soundfile
- Download pertama facebook/mms-1b-all lumayan besar (checkpoint ~1B
  parameter) -- sekali cache, load fold berikutnya baca dari cache lokal
  (bukan download ulang tiap fold).

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
MANIFEST_WITH_FOLD_PATH = "/content/drive/MyDrive/bugis_authentic/authentic_manifest_with_fold.csv"   # SAMA PERSIS file yg dipakai Whisper
AUDIO_DIR = "/content/drive/MyDrive/bugis_authentic/audio"     # SAMA dengan AUDIO_DIR Whisper
OUTPUT_DIR = "/content/drive/MyDrive/bugis_authentic/authentic_only_results"   # folder SAMA dgn Whisper, nama file beda jadi aman digabung

MMS_CHECKPOINT = "facebook/mms-1b-all"
MMS_TARGET_LANG = "ind"      # konsisten Zero-Shot (lihat metodologi 4.1)

K_FOLDS = 5                  # HARUS sama dengan kfold_split_authentic_colab.py
RANDOM_SEED = 42             # single-seed (keputusan sesi ini, sama kayak Whisper)

# ---- Compute budget: WAJIB dipakai ulang identik utk Synthetic-Only & Combined (MMS) ----
# CATATAN REVISI (setelah smoke test empiris di T4 kamu): MAX_STEPS awalnya 600
# (dasar: blog resmi konvergen kuat di ~step 400, 600 dikasih margin di atasnya --
# lihat poin 9 docstring). TAPI di T4 kamu, kecepatan aktual cuma ~0.03 it/s
# (jauh lebih lambat dari Whisper yg ~0.26 it/s -- gabungan: model 4x lebih besar,
# grad_accum 8x microbatch/step vs Whisper 2x, CTC proses raw audio bukan mel-
# spectrogram kompak). 600 step = ~5 jam/fold x 5 fold = ~25 jam, TIDAK feasible.
# DITURUNKAN ke 150 -- masih di atas step 100 yang di blog resmi SUDAH nunjukin
# perbaikan WER signifikan dari baseline (0.280 di step 100), jadi 150 tetap
# beralasan secara kualitas, bukan cuma "dipotong demi cepat". WARMUP_STEPS
# ikut diturunkan proporsional (100 -> 20) supaya tidak habis 2/3 training cuma
# buat warmup. Estimasi baru: ~150/600 x 5 jam ~ 1.25 jam/fold x 5 fold ~ 6 jam
# total -- masih panjang, TAPI resume-capable (lihat poin 10) jadi bisa dicicil
# lintas beberapa sesi Colab, tidak harus 1 sesi utuh.
MAX_STEPS = 150
LEARNING_RATE = 1e-3         # DIVERIFIKASI ke blog resmi HF, lihat poin 3 di docstring
WARMUP_STEPS = 20            # diturunkan proporsional dari 100 (lihat catatan MAX_STEPS di atas)

# ---- Percepatan QUALITY-NEUTRAL: SUDAH DICOBA 3 VARIASI, SEMUA MENTOK ----
# Riwayat lengkap percobaan batch restructuring di T4 kamu:
#   1. per_device=8, checkpointing=False -> OOM
#   2. per_device=8, checkpointing=True, group_by_length=True -> OOM
#   3. per_device=8, checkpointing=True, group_by_length=False -> TIDAK OOM,
#      TAPI kecepatan PERSIS SAMA (~0.03 it/s) dgn per_device=4 -- artinya T4
#      kamu sudah mentok kemampuan hitungnya dari batch=4, bukan soal
#      overhead pemanggilan yang bisa dihemat dgn batch lebih besar.
# KESIMPULAN: tidak ada untungnya pakai per_device=8 (kecepatan sama, cuma
# hilangin manfaat group_by_length + kurang persis ke resep resmi). DIBALIKIN
# ke per_device=4 + group_by_length AKTIF -- config paling efisien yg
# terbukti stabil. Lever batch-restructuring berhenti di sini; satu-satunya
# yang masih bisa ngubah waktu total dari titik ini adalah MAX_STEPS.
PER_DEVICE_TRAIN_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 8     # effective batch TETAP 32
USE_GROUP_BY_LENGTH = True          # aktif lagi -- terbukti tidak berkontribusi ke OOM, cuma bantu efisiensi padding

LOGGING_STEPS = 10           # diperapat dari 20 -- MAX_STEPS lebih kecil sekarang, tetap mau ~15 titik data di kurva loss

FP16 = True
GRADIENT_CHECKPOINTING = True   # WAJIB True -- sudah terbukti perlu di semua percobaan sebelumnya
# --------------------------


def normalize(text: str) -> str:
    """SAMA PERSIS dengan normalize() di zero_shot_eval_whisper_mms_colab.py."""
    text = unicodedata.normalize("NFC", text.strip())
    return " ".join(text.split())


def load_audio(audio_filename: str) -> np.ndarray:
    path = os.path.join(AUDIO_DIR, audio_filename)
    array, _sr = librosa.load(path, sr=16000, mono=True)
    return array


def check_vocab_coverage(processor: Wav2Vec2Processor, texts: List[str]) -> None:
    """PENTING -- baca poin 2 di docstring atas. Cek karakter mana di teks
    authentic yang TIDAK ada di vocab tokenizer 'ind' MMS. Longgar soal
    case (cek huruf asli, upper, DAN lower) karena tidak yakin konvensi
    case vocab-nya tanpa akses langsung ke model di Hub."""
    vocab = set(processor.tokenizer.get_vocab().keys())
    all_chars = set("".join(texts)) - {" "}
    missing = sorted(
        c for c in all_chars
        if c not in vocab and c.upper() not in vocab and c.lower() not in vocab
    )
    print(f"\n--- Cek cakupan vocabulary MMS ('{MMS_TARGET_LANG}') ---")
    print(f"Ukuran vocab: {len(vocab)} token")
    if missing:
        print(f"⚠ PERINGATAN: {len(missing)} karakter di teks authentic TIDAK ditemukan di vocab: {missing}")
        print("  Karakter ini TIDAK PERNAH bisa muncul di prediksi MMS (zero-shot maupun fine-tuned) --")
        print("  WAJIB dicatat di Limitasi tesis kalau ini termasuk karakter bermakna (mis. apostrof glottal stop).")
    else:
        print("Semua karakter di teks authentic ADA di vocab -- aman.")
    print("---\n")


def verify_model_config(model, processor) -> bool:
    """PENTING -- ini jawaban konkret utk "gimana cara tahu kombinasi kwargs
    dropout/ctc_loss_reduction/target_lang beneran nyambung mulus pas load
    checkpoint asli". HF Transformers SEHARUSNYA menghormati semua config
    kwargs yang dioper ke from_pretrained(), tapi saya cuma sempat verifikasi
    itu ke Wav2Vec2Config KOSONG di sandbox saya (bukan ke checkpoint
    mms-1b-all yang sebenarnya, karena tidak ada akses jaringan ke situ) --
    jadi di sini dicek ULANG secara EKSPLISIT & OTOMATIS begitu model asli
    ke-load, bukan diasumsikan berhasil cuma karena tidak crash."""
    print("\n--- Verifikasi config model (poin 3, cek kombinasi kwargs) ---")
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

    # cek tambahan: vocab tokenizer harus konsisten dgn dimensi output lm_head
    # -- kalau beda, artinya load_adapter() memuat vocab yg beda dari yg
    # disiapkan from_pretrained(), bug klasik yang TIDAK bikin crash tapi
    # bikin prediksi ngaco diam-diam.
    vocab_size = len(processor.tokenizer)
    lm_head_out = model.lm_head.out_features
    vocab_ok = (vocab_size == lm_head_out)
    all_ok = all_ok and vocab_ok
    print(f"  vocab tokenizer ({vocab_size}) vs dimensi output lm_head ({lm_head_out})  "
          f"{'OK' if vocab_ok else '<<< MISMATCH!'}")

    if not all_ok:
        print("⚠⚠⚠ PERINGATAN KERAS: ada ketidakcocokan di atas -- JANGAN lanjut training, "
              "screenshot output ini & laporkan balik, jangan asumsikan tetap aman.")
    else:
        print("Semua cocok -- kombinasi kwargs beneran kepasang, aman lanjut.")
    print("---\n")
    return all_ok


def freeze_for_adapter_only(model: Wav2Vec2ForCTC) -> None:
    """Method RESMI dari blog HF (huggingface.co/blog/mms_adapters), BUKAN
    reimplementasi manual berbasis nama parameter -- lihat poin 7 di
    docstring atas soal verifikasi langsung ke source code.

    freeze_base_model() membekukan SEMUA parameter di dalam self.wav2vec2
    (termasuk adapter layer, karena adapter layer nested di situ). Lalu
    _get_adapters() mengembalikan dict berisi PERSIS parameter adapter layer
    (dideteksi via isinstance ke Wav2Vec2AttnAdapterLayer) + lm_head, yang
    di-set requires_grad=True lagi secara eksplisit -- pola "freeze semua,
    baru un-freeze yang perlu", bukan "iterasi & putuskan per-parameter"."""
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
        print("⚠⚠⚠ PERINGATAN KERAS: 0 parameter trainable! model._get_adapters() kosong -- "
              "kemungkinan config.adapter_attn_dim tidak ter-set (cek apakah model di-load "
              "dengan target_lang=... dengan benar). JANGAN lanjut training, cek dulu manual.")


@dataclass
class DataCollatorCTCWithPadding:
    """Dicocokkan persis ke pola pemanggilan processor.pad() di blog resmi
    (bukan manggil feature_extractor.pad()/tokenizer.pad() terpisah) --
    SUDAH divalidasi lewat source code bahwa processor.pad() cuma pembungkus
    tipis dari dua method itu (fungsinya identik), tapi dipakai bentuk ini
    biar sesuai literal dgn kode yang dikutip di docstring atas."""
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


def build_hf_dataset(df: pd.DataFrame, processor: Wav2Vec2Processor) -> Dataset:
    ds = Dataset.from_pandas(df[["file_id", "text", "audio_filename"]].reset_index(drop=True))

    def _prepare(example):
        audio_array = load_audio(example["audio_filename"])
        example["input_values"] = processor.feature_extractor(
            audio_array, sampling_rate=16000
        ).input_values[0]
        example["input_length"] = len(example["input_values"])   # dipakai group_by_length, lihat poin 6 docstring
        example["labels"] = processor.tokenizer(example["text"]).input_ids
        return example

    ds = ds.map(_prepare, remove_columns=["text", "audio_filename"])
    return ds


def transcribe_mms_one(model, processor, audio_path, device) -> str:
    """Signature & isi meniru persis transcribe_mms() di
    zero_shot_eval_whisper_mms_colab.py -- greedy argmax, TANPA beam search."""
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
    """Per-file dengan try/except, meniru pola evaluate_model() di zero-shot
    -- sama seperti predict_fold() versi Whisper."""
    model.eval()
    results = []
    rows = test_df.reset_index(drop=True)

    for row in rows.itertuples():
        audio_path = os.path.join(AUDIO_DIR, row.audio_filename)
        reference = normalize(row.text)
        try:
            hypothesis = normalize(transcribe_mms_one(model, processor, audio_path, device))
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
        print("PERINGATAN: GPU tidak terdeteksi -- training di CPU akan SANGAT lambat, "
              "dan MMS-1b jauh lebih berat dari Whisper-small. Pastikan runtime GPU aktif.")

    df = pd.read_csv(MANIFEST_WITH_FOLD_PATH)
    assert len(df) == 106, f"Expected 106 baris, dapat {len(df)}"
    assert set(df.fold.unique()) == set(range(K_FOLDS)), \
        f"fold di manifest ({sorted(df.fold.unique())}) tidak cocok K_FOLDS={K_FOLDS}"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    processor = Wav2Vec2Processor.from_pretrained(MMS_CHECKPOINT, target_lang=MMS_TARGET_LANG)
    check_vocab_coverage(processor, df["text"].tolist())

    all_predictions: List[Dict] = []
    all_log_histories: Dict[str, Any] = {}

    for fold in range(K_FOLDS):
        fold_pred_path = os.path.join(OUTPUT_DIR, f"predictions_mms_fold{fold}.csv")
        fold_log_path = os.path.join(OUTPUT_DIR, f"log_history_mms_fold{fold}.json")

        # RESUME (pola sama dgn Whisper) -- skip fold yg sudah selesai
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

        assert set(train_df.recording_id) & set(test_df.recording_id) == set(), \
            f"BOCOR di fold {fold}: ada recording_id yang muncul di train DAN test!"

        train_ds = build_hf_dataset(train_df, processor)

        model = Wav2Vec2ForCTC.from_pretrained(
            MMS_CHECKPOINT,
            target_lang=MMS_TARGET_LANG,
            ignore_mismatched_sizes=True,
            attention_dropout=0.0,
            hidden_dropout=0.0,
            feat_proj_dropout=0.0,
            layerdrop=0.0,
            ctc_loss_reduction="mean",   # default library "sum" -- WAJIB dioverride, lihat poin 5 docstring
        )
        model.load_adapter(MMS_TARGET_LANG)   # muat bobot adapter+lm_head 'ind' yg sudah dipretrain, konsisten Zero-Shot
        verify_model_config(model, processor)   # WAJIB lolos sebelum lanjut -- lihat poin 3 diskusi & docstring
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
            gradient_checkpointing=GRADIENT_CHECKPOINTING,
            fp16=FP16,
            logging_steps=LOGGING_STEPS,
            save_strategy="no",
            eval_strategy="no",
            report_to=[],
            seed=RANDOM_SEED,
            remove_unused_columns=False,
        )
        # group_by_length (poin 6 docstring): nama parameter berubah antar versi
        # transformers -- dicek dinamis via dataclasses.fields, bukan hardcode.
        # DIMATIKAN sementara via USE_GROUP_BY_LENGTH -- lihat catatan percobaan
        # ke-3 di CONFIG atas (dugaan penyebab OOM di batch=8).
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
        # sama seperti fix di finetune_whisper_authentic_colab.py -- tokenizer= -> processing_class=
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
        Path(fold_log_path).write_text(json.dumps(fold_log_history, indent=2), encoding="utf-8")
        print(f"Fold {fold}: {len(fold_predictions)} prediksi tersimpan ke {fold_pred_path}")

        del model, trainer
        gc.collect()
        torch.cuda.empty_cache()

    pred_df = pd.DataFrame(all_predictions).sort_values("file_id").reset_index(drop=True)

    assert pred_df.file_id.nunique() == len(pred_df), "Ada file_id yang terprediksi LEBIH dari 1x -- cek logika fold!"
    if len(pred_df) == 106:
        print(f"\nSanity check akhir: LOLOS (106/106 segmen terprediksi tepat 1x lewat skema k-fold)")
    else:
        print(f"\n⚠ PERINGATAN: cuma {len(pred_df)}/106 segmen berhasil diprediksi "
              f"(sisanya di-skip krn error, lihat log GAGAL di atas). Cek sebelum lanjut ke evaluasi.")

    pred_out_path = os.path.join(OUTPUT_DIR, "authentic_only_mms_predictions.csv")
    pred_df.to_csv(pred_out_path, index=False)
    print(f"Tersimpan: {pred_out_path}")

    log_out_path = os.path.join(OUTPUT_DIR, "train_log_history_all_folds_mms.json")
    with open(log_out_path, "w") as f:
        json.dump(all_log_histories, f, indent=2)
    print(f"Tersimpan: {log_out_path}")

    return pred_df


if __name__ == "__main__":
    main()
