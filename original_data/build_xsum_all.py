import json
from datasets import load_dataset
from tqdm import tqdm

OUTPUT_JSON = "xsum_all.json"

dataset = load_dataset("xsum", split="train")

data = []
orig_index = 0

for item in tqdm(dataset):
    data.append({
        "dataset": "xsum",
        "orig_index": orig_index,
        "text": item["document"].strip()
    })
    orig_index += 1

json.dump(data, open(OUTPUT_JSON, "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)

print(f"[OK] Built xsum_all.json with {len(data)} samples.")
