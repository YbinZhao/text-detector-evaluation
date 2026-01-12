import json
import os
import random
import logging
import re
import nltk

HUMAN_PER_DOMAIN = 4000
MAX_WORDS = 512
SEED = 2026

DOMAINS = {
    "arxiv": "arxiv_all.json",
    "xsum": "xsum_all.json",
    "writing_prompt": "writing_prompt_all.json",
    "yelp_review": "yelp_all.json",
    "squad": "squad_all.json"
}

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.getcwd()),
    "Benchmark",
    "TestOnly_Human"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

random.seed(SEED)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    logging.info(f"Saved {len(data)} samples → {path}")


def clean_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\n+", " ", text)
    return text


def word_count(text: str) -> int:
    return len(text.split())


def valid_text(text: str) -> bool:
    """
    Benchmark-stage filtering:
    - non-empty
    - >= 4 sentences
    - <= MAX_WORDS
    """
    if not text:
        return False

    if word_count(text) > MAX_WORDS:
        return False

    try:
        if len(nltk.sent_tokenize(text)) < 4:
            return False
    except Exception:
        return False

    return True


def build_filtered_pool(data, domain):
    pool = []

    for item in data:
        if "text" not in item:
            continue

        text = clean_text(item["text"])
        if not valid_text(text):
            continue

        pool.append({
            "text": text,
            "label": "human",
            "domain": domain,
            "orig_index": item.get("orig_index", None),
            "word_count": word_count(text)
        })

    logging.info(
        f"[{domain}] valid human samples after filtering: {len(pool)}"
    )
    return pool

if __name__ == "__main__":

    for domain, file_name in DOMAINS.items():
        logging.info(f"Processing domain: {domain}")

        data = read_json(file_name)

        pool = build_filtered_pool(
            data=data,
            domain=domain
        )

        assert len(pool) >= HUMAN_PER_DOMAIN, (
            f"[{domain}] Not enough valid samples: "
            f"{len(pool)} < {HUMAN_PER_DOMAIN}"
        )

        sampled = random.sample(pool, HUMAN_PER_DOMAIN)

        out_path = os.path.join(
            OUTPUT_DIR,
            f"{domain}_test.json"
        )

        save_json(sampled, out_path)

    logging.info(
        "All HUMAN TestOnly benchmarks are ready "
        "(filtered + PURE RANDOM sampling)."
    )
