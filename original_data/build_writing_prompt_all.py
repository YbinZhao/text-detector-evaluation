import json
import nltk
from datasets import load_dataset
from tqdm import tqdm

# nltk.download("punkt")

OUTPUT_JSON = "writing_prompt_all.json"

dataset = load_dataset("euclaise/writingprompts", split="train")

def count_sentences(text):
    return len(nltk.sent_tokenize(text))

data = []
orig_index = 0

for item in tqdm(dataset):
    story = item["story"].strip()

    if 800 < len(story) < 2000 and count_sentences(story) >= 4:
        data.append({
            "dataset": "writing",
            "orig_index": orig_index,
            "text": story
        })
        orig_index += 1

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"[OK] Built writing_prompt_all.json with {len(data)} samples.")
