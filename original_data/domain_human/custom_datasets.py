import json
import os
import re
import unicodedata
import logging

INPUT_DIR = r"D:\py\irt-benchmark\Benchmark\TestOnly_Human"
OUTPUT_DIR = r"D:\py\irt-benchmark\Benchmark\TestOnly_Human"

LIMIT_REPEAT = True      # 是否限制极端重复字符
MAX_REPEAT = 5           # 最大重复次数

os.makedirs(OUTPUT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

def normalize_unicode(text: str) -> str:
    """Normalize unicode representations."""
    return unicodedata.normalize("NFKC", text)


def remove_control_chars(text: str) -> str:
    """Remove invisible / control characters."""
    return re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)


def process_spaces(text: str) -> str:
    """Your original normalization logic (unchanged)."""
    return text.replace(
        ' ,', ',').replace(
        ' .', '.').replace(
        ' ?', '?').replace(
        ' !', '!').replace(
        ' ;', ';').replace(
        ' \'', '\'').replace(
        ' ’ ', '\'').replace(
        ' :', ':').replace(
        '<newline>', '\n').replace(
        '`` ', '"').replace(
        ' \'\'', '"').replace(
        '\'\'', '"').replace(
        '.. ', '... ').replace(
        ' )', ')').replace(
        '( ', '(').replace(
        ' n\'t', 'n\'t').replace(
        ' i ', ' I ').replace(
        ' i\'', ' I\'').replace(
        '\\\'', '\'').replace(
        '\n ', '\n').strip()


def normalize_whitespace(text: str) -> str:
    """Collapse multiple whitespaces."""
    return re.sub(r'\s+', ' ', text).strip()


def limit_repeated_chars(text: str, max_repeat=5) -> str:
    """Limit extreme repeated characters (e.g., !!!!!)."""
    return re.sub(r'(.)\1{%d,}' % max_repeat, r'\1' * max_repeat, text)


def final_clean(text: str) -> str:
    """Final safe cleaning pipeline."""
    text = normalize_unicode(text)
    text = remove_control_chars(text)
    text = process_spaces(text)
    text = normalize_whitespace(text)

    if LIMIT_REPEAT:
        text = limit_repeated_chars(text, MAX_REPEAT)

    return text

def clean_file(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cleaned = []
    for item in data:
        new_item = dict(item)  # shallow copy
        new_item["text"] = final_clean(item["text"])
        cleaned.append(new_item)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=4)

    logging.info(
        f"[OK] {os.path.basename(input_path)} → "
        f"{len(cleaned)} samples cleaned"
    )


if __name__ == "__main__":

    for fname in os.listdir(INPUT_DIR):
        if not fname.endswith(".json"):
            continue

        in_path = os.path.join(INPUT_DIR, fname)
        out_path = os.path.join(OUTPUT_DIR, fname)

        clean_file(in_path, out_path)

    logging.info("All sampled benchmarks cleaned successfully.")
