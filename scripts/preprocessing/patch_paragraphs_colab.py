# ============================================================
# Tambal teks yang sudah diperbaiki manual ke file master 201 paragraf
# ============================================================

import csv
from pathlib import Path


MASTER_CSV = "/content/sampel_201_final.csv"          # file 201 paragraf asli
OUTPUT_CSV = "/content/sampel_201_final_fixed.csv"     # hasil setelah ditambal

# source_index (dari sentences_for_tts_no_punctuation.csv) -> teks yang sudah diperbaiki
FIXES = {
    "13763": "Kuburu tuhan Yesus Kristus sedang dibukka. Untu' pertamakalinya eddi mancaji di Yerusalem ketika mereka meluncurkan marmer kuburu tuhan Yesus. Saddang sangkakala diangkalinga na ellung mabbentu lingkaran. Puji tuhan, fenomena yang dena naulle dijelaskang ancaji esso eddu di Israel, like pelik tafi tuju.",
    "16655": "Maladde upaya saling masserang na saling bela keddi mancaji gejala laleng masyarakat Indonesia. Habib Rizieq Syihab ero yang dihenni napimping demo aksi bela Islam, cappu nasaba nabela Al Quran furanna diduga dihinakan pole Ahok. Kin sisumbe elo rifigau aksi bela Yesus, absus furanna ero diaseng mappigau nakkeddai kepada dogma ajaran Kristen okko Desember labe.",
    "67542": "Sibawa yangerangi matengena Kristus ku idi' maneng, idi' maneng odding punnai pikiri na papahange iyya malanre na mapacing, nasaba wettuta maneng pajanengi maraga solina iyya nawaja Yesus untu' mpajaiki manneg. Iyyanaro wettu iyya magello untu' mitai keadaana atie na asilongeta maneng sibawa padata.",
    "28876": "Yolona, ku pamulang aha' iyyaro, iyya mabaca okina Lukas macarita kajajiang ancajingena Yesus, na iyya upajenengi makeda ku esso Natal iyya pamulang iyyaro de' nengka acara, hadida, na anre loppo bangsana iyya idi' maneng pegaui makukuae. Banna iyya manessae, ku rate engka asembange.",
    "128436": "Missenggah idi tofi kerucut taung baru tanda selleng fura murtad, tofi taung baru yg mabbentu kerucut ternyata yanaritu tofi dengan bentu yang di difau sanbenito. Yakni tofi yg difake selleng Andalusia untu natandai narekko alena fura murtad, diaha penindasan gereja Katholik Roma yang menerapkan inkuisisi Spanyol.",
    "103152": "Tau ero dena manessa agamana, mita photo lakkai baine sedang makkatenni persilahkan santapan ulu bahi di ase mejang manre na nacoeri peribadatan digereja. Tau pekkeddi elo diakka jaji gubernur Sumut, bafa Djarot asli selleng ambisi elo jaji gubernur Sumut sedia makkelong koor kebaktian di gereja, tergadai agama nasaba elo jaji pemimping, di tikkenang layar pesang yang nakirinang melalui WhatsApp.",
}
# --------------------------


rows = []
with open(MASTER_CSV, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    id_candidates = ["paragraph_id", "index", "id"]
    id_col = next((c for c in id_candidates if c in fieldnames), fieldnames[0])

    patched_count = 0
    for row in reader:
        row_id = row[id_col]
        if row_id in FIXES:
            if FIXES[row_id].startswith("GANTI DENGAN"):
                print(f"⚠ source_index={row_id}: belum diisi, teks LAMA dipakai (belum diperbaiki)")
            else:
                print(f"✓ source_index={row_id}: ditambal dengan teks baru")
                row["text"] = FIXES[row_id]
                patched_count += 1
        rows.append(row)

with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"\n{patched_count}/{len(FIXES)} paragraf berhasil ditambal.")
print(f"Disimpan ke: {OUTPUT_CSV}")

found_ids = {row[id_col] for row in rows}
missing = [k for k in FIXES if k not in found_ids]
if missing:
    print(f"\n⚠ source_index berikut TIDAK ketemu di {MASTER_CSV}, cek lagi: {missing}")
