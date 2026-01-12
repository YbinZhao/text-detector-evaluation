import json

INPUT_JSON = "train-v1.1.json"
OUTPUT_JSON = "squad_all.json"

with open(INPUT_JSON, "r", encoding="utf-8") as f:
    squad = json.load(f)

data = []
seen = set()
orig_index = 0

for article in squad["data"]:
    for para in article["paragraphs"]:
        context = para["context"].strip()
        if context in seen:
            continue
        seen.add(context)

        data.append({
            "dataset": "squad",
            "orig_index": orig_index,
            "text": context
        })
        orig_index += 1

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"[OK] Built squad_all.json with {len(data)} samples.")
