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
from metrics import get_roc_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def get_entropy(text, args, tokenizer, model):

    if not text:
        return 0.0
        
    # 确保 pad_token 存在
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    tokenized = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(args.DEVICE)
    labels = tokenized.input_ids[:, 1:]
    
    with torch.no_grad():
        outputs = model(**tokenized)
        logits = outputs.logits[:, :-1]
        
        probs = torch.nn.functional.softmax(logits, dim=-1)
        log_probs = torch.log(probs + 1e-9)
        entropy_per_token = -torch.sum(probs * log_probs, dim=-1)
        
        mask = (labels != tokenizer.pad_token_id)
    
        total_entropy = (entropy_per_token * mask).sum()
        token_count = mask.sum()
        
        if token_count == 0:
            return 0.0
            
        avg_entropy = total_entropy / token_count
        
    return avg_entropy.item()

def load_data_flexible(path):
    with open(path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    test_data = []
    if isinstance(raw_data, dict) and "original" in raw_data:
        for text in raw_data["original"]:
            test_data.append({"text": text, "label": "human"})
        for text in raw_data["sampled"]:
            test_data.append({"text": text, "label": "llm"})
    else:
        test_data = raw_data
    return test_data

def get_processed_indices(out_path):
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
    logging.info(f"Loading base model: {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(args.base_model).eval().to(args.DEVICE)

    filenames = args.test_data_path.split(",")
    for filename in filenames:
        logging.info(f"Processing file: {filename}")
        
        test_data = load_data_flexible(filename)
        
        output_stream = filename.split(".json")[0] + "_entropy_stream.jsonl"
        processed_set = get_processed_indices(output_stream)
        logging.info(f"Resuming: {len(processed_set)} items already processed.")

        fout = open(output_stream, "a", encoding="utf-8")

        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        for item in tqdm.tqdm(test_data, desc="Calculating Entropy"):
            if item["text"] in processed_set:
                continue

            text = item["text"]
            
            entropy_score = get_entropy(text, args, tokenizer, model)
            
            record = {
                "text": text,
                "label": item["label"],
                "entropy": entropy_score
            }

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()

        fout.close()

        logging.info("Calculating metrics from stream file...")
        final_data = []
        human_scores = []
        llm_scores = []

        with open(output_stream, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    final_data.append(data)
                    score_for_metric = -data["entropy"]
                    
                    if data["label"] == "human":
                        human_scores.append(score_for_metric)
                    elif data["label"] == "llm":
                        llm_scores.append(score_for_metric)
                except:
                    continue

        human_scores = [i for i in human_scores if np.isfinite(i)]
        llm_scores = [i for i in llm_scores if np.isfinite(i)]

        roc_auc, optimal_threshold, conf_matrix, precision, recall, f1, accuracy, *_ = get_roc_metrics(
            human_scores, llm_scores
        )

        correctness_sequence = []
        for item in final_data:
            score = -item["entropy"]
            label = item["label"]
            
            pred_label = "llm" if score >= optimal_threshold else "human"
            is_correct = 1 if pred_label == label else 0
            
            correctness_sequence.append(is_correct)
            
            item["prediction"] = pred_label
            item["is_correct"] = bool(is_correct)
            item["neg_entropy_score"] = score

        seq_as_str = "".join(map(str, correctness_sequence))
        
        logging.info(f"ROC AUC: {roc_auc:.4f} | Optimal Threshold (-Entropy): {optimal_threshold:.4f}")
        logging.info(f"Correctness Sequence Length: {len(seq_as_str)}")

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

        data_out = filename.split(".json")[0] + "_entropy_data.json"
        with open(data_out, "w", encoding="utf-8") as f:
            json.dump(final_data, f, indent=4, ensure_ascii=False)

        res_out = filename.split(".json")[0] + "_entropy_result.json"
        with open(res_out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
            
        logging.info(f"Done. Results saved to {res_out}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data_path', type=str, required=True,
                        help="Path to the test data json file.")
    parser.add_argument('--base_model', default="EleutherAI/gpt-neo-2.7B", type=str, required=False)
    parser.add_argument('--DEVICE', default="cuda", type=str, required=False)
    parser.add_argument('--seed', default=2026, type=int, required=False)
    args = parser.parse_args()
    
    if not hasattr(args, "DEVICE"):
        args.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    experiment(args)