# -*- coding: utf-8 -*-
import logging
import random
import numpy as np
import torch
import tqdm
import argparse
import json
import os
import sys

from transformers import AutoTokenizer, AutoModelForCausalLM
from rank import get_rank
from metrics import get_roc_metrics

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

def load_data_flexible(path):
    logging.info(f"正在读取文件: {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    
    test_data = []
    if isinstance(raw_data, dict) and "original" in raw_data:
        logging.info("检测到字典格式数据集，正在进行平铺转换...")
        for t in raw_data["original"]:
            test_data.append({"text": t, "label": "human"})
        for t in raw_data["sampled"]:
            test_data.append({"text": t, "label": "llm"})
    elif isinstance(raw_data, list):
        test_data = raw_data
    else:
        raise ValueError("未知的数据集格式，请确保是列表或带有 original/sampled 键的字典。")
    
    return test_data

def get_processed_texts(out_path):
    processed = set()
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    processed.add(data["text"])
                except:
                    continue
    return processed

def experiment(args):
    logging.info(f"正在加载模型: {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(args.base_model).eval().to(args.DEVICE)

    filenames = args.test_data_path.split(",")
    for filename in filenames:
        logging.info(f"处理文件: {filename}")
        
        test_data = load_data_flexible(filename)
        
        output_stream = filename.split(".json")[0] + "_rank_stream.jsonl"
        processed_set = get_processed_texts(output_stream)
        logging.info(f"续传检查: 已处理 {len(processed_set)} 条。")

        fout = open(output_stream, "a", encoding="utf-8")

        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        for item in tqdm.tqdm(test_data, desc="Evaluating Rank"):
            if item["text"] in processed_set:
                continue

            text = item["text"]
            rank_val = get_rank(text, args, tokenizer, model, log=False)
            score = -rank_val if rank_val is not None else -1e9

            record = {
                "text": text,
                "label": item["label"],
                "text_rank_score": score
            }

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()

        fout.close()

        logging.info("正在从流文件计算最终指标...")
        final_data = []
        human_scores = []
        llm_scores = []

        with open(output_stream, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                final_data.append(data)
                if data["label"] == "human":
                    human_scores.append(data["text_rank_score"])
                else:
                    llm_scores.append(data["text_rank_score"])

        res_metrics = get_roc_metrics(human_scores, llm_scores)
        roc_auc, optimal_threshold, conf_matrix, precision, recall, f1, accuracy = res_metrics[:7]

        correctness_sequence = []
        for item in final_data:
            score = item["text_rank_score"]
            pred_label = "llm" if score >= optimal_threshold else "human"
            is_correct = 1 if pred_label == item["label"] else 0
            
            correctness_sequence.append(is_correct)
            item["prediction"] = pred_label
            item["is_correct"] = bool(is_correct)

        seq_as_str = "".join(map(str, correctness_sequence))
        
        result = {
            "roc_auc": roc_auc,
            "optimal_threshold": optimal_threshold,
            "conf_matrix": conf_matrix,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
            "correctness_sequence": seq_as_str
        }

        logging.info(f"结果: AUC={roc_auc:.4f}, ACC={accuracy:.4f}")

        with open(filename.split(".json")[0] + "_rank_data.json", "w", encoding="utf-8") as f:
            json.dump(final_data, f, indent=4, ensure_ascii=False)

        with open(filename.split(".json")[0] + "_rank_result.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data_path', type=str, required=True)
    parser.add_argument('--base_model', default="EleutherAI/gpt-neo-2.7B", type=str)
    parser.add_argument('--DEVICE', default="cuda", type=str)
    parser.add_argument('--seed', default=2026, type=int)
    args = parser.parse_args()
    
    if args.DEVICE == "cuda" and not torch.cuda.is_available():
        args.DEVICE = "cpu"

    experiment(args)