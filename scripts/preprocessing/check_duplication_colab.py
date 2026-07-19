# ============================================================
# Check duplication — versi Colab (tanpa argparse, tinggal edit config di bawah lalu Run)
# ============================================================

import csv
import unicodedata
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

# ---- EDIT BAGIAN INI ----
MANIFEST_PATH = "/content/synthetic_manifest.csv"   # hasil dari build_manifest_colab.py
ID_COL = "file_id"
TEXT_COL = "text"
OUTPUT_PREFIX = "/content/dup_report"
NEAR_DUP_THRESHOLD = 0.90
SKIP_NEAR_DUP = False          # True = lebih cepat, skip pengecekan near-duplicate
EXPECTED_GROUP_SIZE = 2        # 2 karena skema ardi/gadis (tiap paragraf sengaja 2x)
# --------------------------


def normalize_for_comparison(text, casefold=True):
    text = unicodedata.normalize("NFC", text.strip())
    text = " ".join(text.split())
    return text.casefold() if casefold else text


def load_manifest(path, id_col, text_col):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if id_col not in row or text_col not in row:
                raise ValueError(
                    f"Manifest missing expected columns. Found: {list(row.keys())}. "
                    f"Expected id_col='{id_col}', text_col='{text_col}'."
                )
            rows.append({"file_id": row[id_col], "text": row[text_col]})
    return rows


def find_exact_duplicates(rows):
    groups = defaultdict(list)
    for row in rows:
        key = normalize_for_comparison(row["text"])
        groups[key].append(row["file_id"])
    return {k: v for k, v in groups.items() if len(v) > 1}


def find_near_duplicates(rows, threshold=0.90):
    normalized = [(row["file_id"], normalize_for_comparison(row["text"]), row["text"]) for row in rows]
    near_dupes = []
    for i in range(len(normalized)):
        id_a, norm_a, orig_a = normalized[i]
        for j in range(i + 1, len(normalized)):
            id_b, norm_b, orig_b = normalized[j]
            if norm_a == norm_b:
                continue
            ratio = SequenceMatcher(None, norm_a, norm_b).ratio()
            if ratio >= threshold:
                near_dupes.append((id_a, id_b, round(ratio, 3), orig_a, orig_b))
    return sorted(near_dupes, key=lambda x: -x[2])


def compute_diversity_stats(rows):
    normalized_texts = [normalize_for_comparison(row["text"]) for row in rows]
    unique_texts = set(normalized_texts)
    all_words = []
    for text in normalized_texts:
        all_words.extend(text.split())
    unique_words = set(all_words)
    return {
        "total_files": len(rows),
        "unique_sentences": len(unique_texts),
        "duplication_rate": round(1 - len(unique_texts) / len(rows), 4) if rows else 0,
        "total_word_tokens": len(all_words),
        "unique_word_types": len(unique_words),
        "type_token_ratio": round(len(unique_words) / len(all_words), 4) if all_words else 0,
        "avg_sentence_length_words": round(len(all_words) / len(rows), 1) if rows else 0,
    }


def analyze_group_sizes(rows, expected_size):
    all_groups = defaultdict(list)
    for row in rows:
        key = normalize_for_comparison(row["text"])
        all_groups[key].append(row["file_id"])
    expected_groups = {k: v for k, v in all_groups.items() if len(v) == expected_size}
    anomalous_groups = {k: v for k, v in all_groups.items() if len(v) != expected_size}
    return {"expected_groups": expected_groups, "anomalous_groups": anomalous_groups}


def write_report(output_prefix, exact_dupes, near_dupes, stats):
    stats_path = f"{output_prefix}_summary.txt"
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write("=== Diversity Summary ===\n")
        for k, v in stats.items():
            f.write(f"{k}: {v}\n")
        f.write(f"\nExact-duplicate groups: {len(exact_dupes)}\n")
        f.write(f"Files involved in exact duplication: {sum(len(v) for v in exact_dupes.values())}\n")
        f.write(f"Near-duplicate pairs: {len(near_dupes)}\n")

    exact_path = f"{output_prefix}_exact_duplicates.csv"
    with open(exact_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["normalized_text", "file_ids", "count"])
        for text, ids in sorted(exact_dupes.items(), key=lambda x: -len(x[1])):
            writer.writerow([text, ";".join(ids), len(ids)])

    near_path = f"{output_prefix}_near_duplicates.csv"
    with open(near_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file_id_a", "file_id_b", "similarity", "text_a", "text_b"])
        for id_a, id_b, ratio, text_a, text_b in near_dupes:
            writer.writerow([id_a, id_b, ratio, text_a, text_b])

    return stats_path, exact_path, near_path


# ---- RUN ----
rows = load_manifest(MANIFEST_PATH, ID_COL, TEXT_COL)
print(f"Loaded {len(rows)} entries from {MANIFEST_PATH}")

stats = compute_diversity_stats(rows)
exact_dupes = find_exact_duplicates(rows)

near_dupes = []
if not SKIP_NEAR_DUP:
    print(f"Running near-duplicate pass (threshold={NEAR_DUP_THRESHOLD})... may take a moment for {len(rows)} entries.")
    near_dupes = find_near_duplicates(rows, threshold=NEAR_DUP_THRESHOLD)

stats_path, exact_path, near_path = write_report(OUTPUT_PREFIX, exact_dupes, near_dupes, stats)

print("\n=== Summary ===")
for k, v in stats.items():
    print(f"  {k}: {v}")
print(f"\nExact-duplicate groups: {len(exact_dupes)} ({sum(len(v) for v in exact_dupes.values())} files involved)")
print(f"Near-duplicate pairs: {len(near_dupes)}")
print(f"\nReports written:\n  {stats_path}\n  {exact_path}\n  {near_path}")

if EXPECTED_GROUP_SIZE > 1:
    analysis = analyze_group_sizes(rows, EXPECTED_GROUP_SIZE)
    n_expected = len(analysis["expected_groups"])
    n_anomalous = len(analysis["anomalous_groups"])
    n_missing = sum(1 for v in analysis["anomalous_groups"].values() if len(v) < EXPECTED_GROUP_SIZE)
    n_excess = n_anomalous - n_missing

    print(f"\n=== Augmentation-aware analysis (expected group size = {EXPECTED_GROUP_SIZE}) ===")
    print(f"  Groups matching expected size {EXPECTED_GROUP_SIZE} (working as designed): {n_expected}")
    print(f"  Groups with FEWER copies than expected (likely a missing voice): {n_missing}")
    print(f"  Groups with MORE copies than expected (genuine unexpected duplication): {n_excess}")

    if n_anomalous:
        anomalous_path = f"{OUTPUT_PREFIX}_anomalous_groups.csv"
        with open(anomalous_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["normalized_text", "file_ids", "count", "expected_count", "issue"])
            for text, ids in sorted(analysis["anomalous_groups"].items(), key=lambda x: len(x[1])):
                issue = "missing_voice(s)" if len(ids) < EXPECTED_GROUP_SIZE else "unexpected_extra"
                writer.writerow([text, ";".join(ids), len(ids), EXPECTED_GROUP_SIZE, issue])
        print(f"  -> Details written to {anomalous_path}")

    print(f"\n  Note: the {n_expected * EXPECTED_GROUP_SIZE} files in expected groups are NOT a data "
          f"quality issue — that's the {EXPECTED_GROUP_SIZE}-way augmentation working as intended. "
          f"Only the anomalous groups need investigation.")
elif stats["duplication_rate"] > 0.05:
    print(f"\n⚠ Duplication rate {stats['duplication_rate']:.1%} — worth reporting explicitly in the thesis.")
