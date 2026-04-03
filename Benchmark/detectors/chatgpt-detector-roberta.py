# -*- coding: utf-8 -*-
import json
import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
import logging

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_args():
    parser = argparse.ArgumentParser(description="ChatGPT Detector Evaluation Script")

    parser.add_argument(
        '--test_data_path',
        type=str,
        required=True,
        help="Path to JSON dataset (must contain 'original' and 'sampled')"
    )
    parser.add_argument(
        '--base_model',
        type=str,
        default="Hello-SimpleAI/chatgpt-detector-roberta",
        help="Pretrained detector model name or path"
    )
    parser.add_argument(
        '--DEVICE',
        type=str,
        default="cuda",
        help="Device for computation ('cuda' or 'cpu')"
    )
    parser.add_argument('--seed', type=int, default=2026, help="Random seed")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for inference")
    parser.add_argument('--max_length', type=int, default=512, help="Max token length per text")
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help="Probability threshold for AI detection"
    )

    return parser.parse_args()

def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    
    args = get_args()
    set_seed(args.seed)

    device = torch.device(
        args.DEVICE if torch.cuda.is_available() and args.DEVICE == "cuda" else "cpu"
    )
    logging.info(f"使用设备: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(args.base_model)
    model.to(device)
    model.eval()

    with open(args.test_data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    assert "original" in raw_data and "sampled" in raw_data, \
        "JSON 必须包含 'original' 和 'sampled' 字段"

    original_texts = raw_data["original"]
    sampled_texts = raw_data["sampled"]

    texts = original_texts + sampled_texts
    y_true = [0] * len(original_texts) + [1] * len(sampled_texts)

    # --- 推理
    y_score = []
    y_pred = []
    results = []

    for i in range(0, len(texts), args.batch_size):
        batch_texts = texts[i:i + args.batch_size]

        enc = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            logits = model(**enc).logits
            probs = F.softmax(logits, dim=-1)

        for j, p in enumerate(probs):
            prob_human = p[0].item()
            prob_ai = p[1].item()
            pred = 1 if prob_ai >= args.threshold else 0

            idx = i + j
            y_score.append(prob_ai)
            y_pred.append(pred)

            results.append({
                "text": texts[idx],
                "label": y_true[idx],
                "pred_label": pred,
                "prob_human": prob_human,
                "prob_ai": prob_ai
            })

    correctness_sequence = [int(p == t) for p, t in zip(y_pred, y_true)]
    seq_as_str = "".join(map(str, correctness_sequence))

    roc_auc = roc_auc_score(y_true, y_score)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = np.mean(correctness_sequence)

    result_summary = {
        "roc_auc": roc_auc,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "correctness_sequence": seq_as_str
    }

    input_path = args.test_data_path
    base_name = input_path[:-5] if input_path.endswith(".json") else input_path
    data_out = f'{base_name}_chatgpt-detector-roberta_data.json'
    res_out = f'{base_name}_chatgpt-detector-roberta_result.json'

    with open(data_out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    with open(res_out, "w", encoding="utf-8") as f:
        json.dump(result_summary, f, indent=4, ensure_ascii=False)

    logging.info("-" * 50)
    logging.info(f"评估完成！统计指标如下:")
    logging.info(f"Accuracy : {accuracy:.4f}")
    logging.info(f"Precision: {precision:.4f}")
    logging.info(f"Recall   : {recall:.4f}")
    logging.info(f"F1 Score : {f1:.4f}")
    logging.info(f"ROC AUC  : {roc_auc:.4f}")
    logging.info("-" * 50)
    logging.info(f"详细数据保存至: {data_out}")
    logging.info(f"统计结果保存至: {res_out}")


if __name__ == "__main__":
    main()
