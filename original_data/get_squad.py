import json
from sampler import sample

INPUT_JSON = "squad_all.json"
OUTPUT_JSON = "squad_4000.json"

with open(INPUT_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

sampled = sample(
    data=data,
    text_key="text",
    n_samples=4000,
    seed=2023
)

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(sampled, f, ensure_ascii=False, indent=2)

print(f"[OK] Saved {len(sampled)} samples to squad_4000.json")
