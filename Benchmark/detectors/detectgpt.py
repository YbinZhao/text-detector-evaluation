# -*- coding: utf-8 -*-
import os
import json
import argparse
import math
import random
import numpy as np
import torch
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM
)
import warnings
warnings.filterwarnings("ignore")

# ======================================================
# 0. 全局配置
# ======================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN = 512

# DetectGPT 判决阈值（论文量级，工程中可再标定）
DETECTGPT_THRESHOLD = 0.1   # score > threshold → machine

ce_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

# ======================================================
# 1. HF cache → snapshot 解析（只给 base 模型用）
# ======================================================
def resolve_snapshot_path(hf_repo_dir: str) -> str:
    snapshots_dir = os.path.join(hf_repo_dir, "snapshots")
    if not os.path.isdir(snapshots_dir):
        raise FileNotFoundError(f"[HF cache] snapshots not found: {snapshots_dir}")

    snaps = [
        os.path.join(snapshots_dir, d)
        for d in os.listdir(snapshots_dir)
        if os.path.isdir(os.path.join(snapshots_dir, d))
    ]
    if not snaps:
        raise FileNotFoundError(f"[HF cache] no snapshots under: {snapshots_dir}")

    snaps.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return snaps[0]

# ======================================================
# 2. DetectGPT：log-likelihood
# ======================================================
@torch.no_grad()
def get_ll(text, tokenizer, model):
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LEN
    ).to(DEVICE)

    logits = model(**enc).logits
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = enc.input_ids[:, 1:].contiguous()
    shift_mask = enc.attention_mask[:, 1:].contiguous()

    loss = ce_loss_fn(
        shift_logits.transpose(1, 2),
        shift_labels
    )
    ll = -(loss * shift_mask).sum() / shift_mask.sum()
    return ll.item()


@torch.no_grad()
def get_lls(texts, tokenizer, model):
    return [get_ll(t, tokenizer, model) for t in texts]

# ======================================================
# 3. DetectGPT 扰动（span infilling）
# ======================================================
def tokenize_and_mask(text, span_length, buffer_size, pct):
    tokens = text.split(" ")
    mask_string = "<<<mask>>>"

    n_spans = int(pct * len(tokens) / (span_length + 2 * buffer_size))
    n_masks = 0

    while n_masks < n_spans and len(tokens) >= span_length:
        start = random.randint(0, max(0, len(tokens) - span_length))
        end = start + span_length
        s = max(0, start - buffer_size)
        e = min(len(tokens), end + buffer_size)
        if mask_string not in tokens[s:e]:
            tokens[start:end] = [mask_string]
            n_masks += 1

    filled = 0
    for i, t in enumerate(tokens):
        if t == mask_string:
            tokens[i] = f"<extra_id_{filled}>"
            filled += 1

    return " ".join(tokens), filled


def extract_fills(text):
    import re
    pattern = re.compile(r"<extra_id_\d+>")
    text = text.replace("<pad>", "").replace("</s>", "").strip()
    return [x.strip() for x in pattern.split(text)[1:-1]]


@torch.no_grad()
def perturb_text(
    text,
    mask_tokenizer,
    mask_model,
    span_length,
    buffer_size,
    pct_words_masked,
    mask_top_p
):
    masked, n_masks = tokenize_and_mask(
        text, span_length, buffer_size, pct_words_masked
    )
    if n_masks == 0:
        return text

    stop_id = mask_tokenizer.encode(f"<extra_id_{n_masks}>")[0]
    enc = mask_tokenizer(masked, return_tensors="pt").to(DEVICE)

    out = mask_model.generate(
        **enc,
        max_length=150,
        do_sample=True,
        top_p=mask_top_p,
        eos_token_id=stop_id
    )

    decoded = mask_tokenizer.decode(out[0], skip_special_tokens=False)
    fills = extract_fills(decoded)

    tokens = masked.split(" ")
    for i in range(n_masks):
        try:
            idx = tokens.index(f"<extra_id_{i}>")
            tokens[idx] = fills[i]
        except Exception:
            return text

    return " ".join(tokens)

# ======================================================
# 4. 主流程
# ======================================================
def main(args):
    # -------- base 模型：只用本地 HF cache --------
    BASE_MODEL_PATH = resolve_snapshot_path(args.base_model_dir)
    print("Base model snapshot:", BASE_MODEL_PATH)

    base_tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_PATH, local_files_only=True
    )
    if base_tokenizer.pad_token is None:
        base_tokenizer.pad_token = base_tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto",
        local_files_only=True
    ).eval()

    # -------- mask 模型：t5-small（自动下载） --------
    print("Loading mask model: t5-small (auto download if needed)")
    mask_tokenizer = AutoTokenizer.from_pretrained("t5-small")
    if mask_tokenizer.pad_token is None:
        mask_tokenizer.pad_token = mask_tokenizer.eos_token

    mask_model = AutoModelForSeq2SeqLM.from_pretrained(
        "t5-small",
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto"
    ).eval()

    # -------- load data --------
    with open(args.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = []

    for item in tqdm(data, desc="DetectGPT"):
        text = item["text"]
        dataset = item.get("dataset", "")
        orig_index = item.get("orig_index", None)

        # perturb
        perturbed = [
            perturb_text(
                text,
                mask_tokenizer,
                mask_model,
                args.span_length,
                args.buffer_size,
                args.pct_words_masked,
                args.mask_top_p
            )
            for _ in range(args.n_perturbation)
        ]

        # likelihood
        text_ll = get_ll(text, base_tokenizer, base_model)
        pert_lls = get_lls(perturbed, base_tokenizer, base_model)
        pert_lls = [x for x in pert_lls if np.isfinite(x)]

        if len(pert_lls) == 0:
            score = 0.0
        else:
            mu = np.mean(pert_lls)
            sigma = np.std(pert_lls) if len(pert_lls) > 1 else 1.0
            score = (text_ll - mu) / sigma

        prediction = "machine" if score > DETECTGPT_THRESHOLD else "human"

        results.append({
            "dataset": dataset,
            "orig_index": orig_index,
            "crit": float(score),
            "prob_machine": None,
            "prediction": prediction
        })

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"[OK] DetectGPT results saved to {args.output_json}")

# ======================================================
# 5. CLI
# ======================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_json", required=True)
    parser.add_argument("--output_json", required=True)

    # base 模型：HF cache repo 目录（与你 FastDetectGPT 一致）
    parser.add_argument(
        "--base_model_dir",
        default=r"D:\cache\models--EleutherAI--gpt-neo-2.7B",
        help="HF cache repo dir for GPT-Neo-2.7B"
    )

    parser.add_argument("--n_perturbation", type=int, default=20)
    parser.add_argument("--span_length", type=int, default=2)
    parser.add_argument("--buffer_size", type=int, default=1)
    parser.add_argument("--pct_words_masked", type=float, default=0.3)
    parser.add_argument("--mask_top_p", type=float, default=1.0)

    args = parser.parse_args()
    main(args)
