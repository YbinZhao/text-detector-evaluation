import json
from datasets import load_dataset
from tqdm import tqdm

OUTPUT_JSON = "arxiv_all.json"

# HuggingFace 官方 arXiv 摘要数据集
dataset = load_dataset("scientific_papers", "arxiv", split="train")

data = []
orig_index = 0

for item in tqdm(dataset):
    abstract = item["abstract"].strip()
    if not abstract:
        continue

    data.append({
        "dataset": "arxiv",
        "orig_index": orig_index,
        "text": abstract
    })
    orig_index += 1

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"[OK] Built arxiv_all.json with {len(data)} samples.")
