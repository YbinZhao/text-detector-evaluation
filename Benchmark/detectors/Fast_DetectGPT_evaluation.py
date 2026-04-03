# -*- coding: utf-8 -*-

import logging
import random
import numpy as np
import torch
import argparse
import json
import os
import sys
from tqdm import tqdm
from metrics import get_roc_metrics
from transformers import AutoTokenizer, AutoModelForCausalLM

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def get_sampling_discrepancy_analytic(logits_ref, logits_score, labels):
    assert logits_ref.shape[0] == 1
    assert logits_score.shape[0] == 1
    assert labels.shape[0] == 1
    if logits_ref.size(-1) != logits_score.size(-1):
        vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
        logits_ref = logits_ref[:, :, :vocab_size]
        logits_score = logits_score[:, :, :vocab_size]

    labels = labels.unsqueeze(-1) if labels.ndim == logits_score.ndim - 1 else labels
    lprobs_score = torch.log_softmax(logits_score, dim=-1)
    probs_ref = torch.softmax(logits_ref, dim=-1)
    log_likelihood = lprobs_score.gather(dim=-1, index=labels).squeeze(-1)
    mean_ref = (probs_ref * lprobs_score).sum(dim=-1)
    var_ref = (probs_ref * torch.square(lprobs_score)).sum(dim=-1) - torch.square(mean_ref)
    discrepancy = (log_likelihood.sum(dim=-1) - mean_ref.sum(dim=-1)) / var_ref.sum(dim=-1).sqrt()
    discrepancy = discrepancy.mean()
    return discrepancy.item()

def get_text_crit(text, args, model_config):
    tokenized = model_config["scoring_tokenizer"](text, return_tensors="pt", return_token_type_ids=False).to(args.DEVICE)
    labels = tokenized.input_ids[:, 1:]
    
    with torch.no_grad():
        logits_score = model_config["scoring_model"](**tokenized).logits[:, :-1]
        
        if args.reference_model == args.scoring_model:
            logits_ref = logits_score
        else:
            tokenized_ref = model_config["reference_tokenizer"](text, return_tensors="pt", return_token_type_ids=False).to(args.DEVICE)
            assert torch.all(tokenized_ref.input_ids[:, 1:] == labels), "Tokenizer mismatch between scoring and reference models"
            logits_ref = model_config["reference_model"](**tokenized_ref).logits[:, :-1]
        
        text_crit = get_sampling_discrepancy_analytic(logits_ref, logits_score, labels)
    return text_crit

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
    logging.info(f"Loading scoring model: {args.scoring_model}...")
    scoring_tokenizer = AutoTokenizer.from_pretrained(args.scoring_model)
    scoring_model = AutoModelForCausalLM.from_pretrained(args.scoring_model).eval().to(args.DEVICE)

    if args.reference_model != args.scoring_model:
        logging.info(f"Loading reference model: {args.reference_model}...")
        reference_tokenizer = AutoTokenizer.from_pretrained(args.reference_model)
        reference_model = AutoModelForCausalLM.from_pretrained(args.reference_model).eval().to(args.DEVICE)
    else:
        logging.info("Reference model is the same as Scoring model (Shared).")
        reference_tokenizer = scoring_tokenizer
        reference_model = scoring_model

    model_config = {
        "reference_tokenizer": reference_tokenizer,
        "reference_model": reference_model,
        "scoring_tokenizer": scoring_tokenizer,
        "scoring_model": scoring_model,
    }

    filenames = args.test_data_path.split(",")
    for filename in filenames:
        logging.info(f"Processing file: {filename}")
        raw_data = json.load(open(filename, "r"))

        test_data = []
        if isinstance(raw_data, dict) and "original" in raw_data:
            for text in raw_data["original"]:
                test_data.append({"text": text, "label": "human"})
            for text in raw_data["sampled"]:
                test_data.append({"text": text, "label": "llm"})
        else:
            test_data = raw_data

        output_jsonl = filename.split(".json")[0] + "_fast_stream.jsonl"
        processed_set = get_processed_texts(output_jsonl)
        logging.info(f"Resuming: {len(processed_set)} items already processed.")

        fout = open(output_jsonl, "a", encoding="utf-8")

        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        for item in tqdm(test_data, desc="Fast-DetectGPT Stream"):
            if item["text"] in processed_set:
                continue
            
            text_crit = get_text_crit(item["text"], args, model_config)
            
            record = {
                "text": item["text"],
                "label": item["label"],
                "text_crit": text_crit
            }

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()
        
        fout.close()

        logging.info("Calculating metrics from stream file...")
        final_data = []
        human_scores = []
        llm_scores = []

        with open(output_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    final_data.append(data)
                    if data["label"] == "human":
                        human_scores.append(data["text_crit"])
                    elif data["label"] == "llm":
                        llm_scores.append(data["text_crit"])
                except:
                    continue
        
        human_scores = [i for i in human_scores if np.isfinite(i)]
        llm_scores = [i for i in llm_scores if np.isfinite(i)]

        roc_auc, optimal_threshold, conf_matrix, precision, recall, f1, accuracy, tpr_001 = get_roc_metrics(
            human_scores, llm_scores)

        correctness_sequence = []
        for item in final_data:
            score = item['text_crit']
            prediction = "llm" if score >= optimal_threshold else "human"
            is_correct = 1 if prediction == item["label"] else 0
            
            correctness_sequence.append(is_correct)
            
            item["prediction"] = prediction
            item["is_correct"] = bool(is_correct)

        seq_as_str = "".join(map(str, correctness_sequence))
        logging.info(f"ROC_AUC: {roc_auc:.4f} | Accuracy: {accuracy:.4f}")

        result = {
            "roc_auc": roc_auc,
            "optimal_threshold": optimal_threshold,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tpr_at_fpr_0_01": tpr_001,
            "correctness_sequence": seq_as_str 
        }

        data_output = filename.split(".json")[0] + "_Fast_DetectGPT_data.json"
        with open(data_output, "w", encoding="utf-8") as f:
            json.dump(final_data, f, indent=4, ensure_ascii=False)

        result_output = filename.split(".json")[0] + "_Fast_DetectGPT_result.json"
        with open(result_output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)

        logging.info(f"Results saved to: {result_output}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data_path', type=str, required=True, help="Input json file")
    parser.add_argument('--reference_model', type=str, default="EleutherAI/gpt-neo-2.7B")
    parser.add_argument('--scoring_model', type=str, default="EleutherAI/gpt-neo-2.7B")
    parser.add_argument('--DEVICE', default="cuda", type=str, required=False)
    parser.add_argument('--seed', default=2026, type=int, required=False)
    args = parser.parse_args()

    if not hasattr(args, "DEVICE"):
        args.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    experiment(args)