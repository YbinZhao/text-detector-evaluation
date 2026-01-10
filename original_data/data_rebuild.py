import json
import os

MAX_WORDS = 512
TEXT_KEY = "text"


def process_spaces(text: str) -> str:
    return (
        text.replace(' ,', ',')
            .replace(' .', '.')
            .replace(' ?', '?')
            .replace(' !', '!')
            .replace(' ;', ';')
            .replace(' \'', '\'')
            .replace(' ’ ', '\'')
            .replace(' :', ':')
            .replace('<newline>', '\n')
            .replace('`` ', '"')
            .replace(' \'\'', '"')
            .replace('\'\'', '"')
            .replace('.. ', '... ')
            .replace(' )', ')')
            .replace('( ', '(')
            .replace(' n\'t', 'n\'t')
            .replace(' i ', ' I ')
            .replace(' i\'', ' I\'')
            .replace('\\\'', '\'')
            .replace('\n ', '\n')
            .strip()
    )


def truncate_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def clean_file(input_path: str, output_path: str):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert isinstance(data, list), "JSON must be a list"

    for item in data:
        text = item[TEXT_KEY]
        text = process_spaces(text)
        text = truncate_words(text, MAX_WORDS)

        item[TEXT_KEY] = text
        item["word_count"] = len(text.split())

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"[OK] {os.path.basename(input_path)} → cleaned")


if __name__ == "__main__":
    files = [
        "arxiv_4000.json",
        "squad_4000.json",
        "writing_4000.json",
        "xsum_4000.json",
        "yelp_4000.json"
    ]

    for fn in files:
        out = fn.replace(".json", ".clean.json")
        clean_file(fn, out)
