# -*- coding: utf-8 -*-
"""
Count text length distribution for arxiv_all.json
Length bins: 0-19, 20-39, ..., 340-359, >=360
"""

import json
from collections import defaultdict, OrderedDict
from pathlib import Path


# ===================== 配置区 =====================
INPUT_JSON = r"D:\py\irt-benchmark\original_data\arxiv_all.json"
BIN_SIZE = 20
MAX_WORDS = 360
EXPORT_CSV = True
OUTPUT_CSV = "arxiv_length_stats.csv"
# =================================================


def word_count(text: str) -> int:
    return len(text.split())


def main():
    input_path = Path(INPUT_JSON)
    assert input_path.exists(), f"File not found: {INPUT_JSON}"

    bins = list(range(0, MAX_WORDS, BIN_SIZE)) + [10**9]

    buckets = defaultdict(int)

    # load data
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # count
    for item in data:
        wc = word_count(item["text"])
        for b in bins:
            if wc < b + BIN_SIZE:
                buckets[b] += 1
                break

    # sort bins
    buckets = OrderedDict(sorted(buckets.items()))

    # print result
    print("=" * 60)
    print("Text Length Distribution (word-count bins)")
    print("=" * 60)

    total = 0
    for b, cnt in buckets.items():
        if b == 10**9:
            label = ">=360"
        else:
            label = f"{b:3d}-{b + BIN_SIZE - 1:3d}"
        print(f"{label:>10} : {cnt}")
        total += cnt

    print("-" * 60)
    print(f"Total samples: {total}")
    print("=" * 60)

    # optional: export CSV
    if EXPORT_CSV:
        import csv
        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["len_bin", "range", "count"])
            for b, cnt in buckets.items():
                if b == 10**9:
                    r = ">=360"
                else:
                    r = f"{b}-{b + BIN_SIZE - 1}"
                writer.writerow([b, r, cnt])
        print(f"[OK] Length stats saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
