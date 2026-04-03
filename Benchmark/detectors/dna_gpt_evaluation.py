# -*- coding: utf-8 -*-
import logging
import random
import re
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


def clean_text(text):
    text = re.sub(r'\n', r'', text)
    return text

def truncate_text_to_sentences(text, min_word_count=100):
    word_count = 0
    end_of_last_sentence = 0
    words = text.split()
    for i, word in enumerate(words):
        if word[-1] in '.!?':
            if word_count >= min_word_count:
                return ' '.join(words[:i + 1])
            end_of_last_sentence = i
        word_count += 1
    if end_of_last_sentence > 0:
        return ' '.join(words[:end_of_last_sentence + 1])
    else:
        return ' '.join(words)

def get_likelihood(logits, labels, pad_index):
    labels = labels.unsqueeze(-1) if labels.ndim == logits.ndim - 1 else labels
    lprobs = torch.log_softmax(logits, dim=-1)
    log_likelihood = lprobs.gather(dim=-1, index=labels)
    mask = labels != pad_index
    log_likelihood = (log_likelihood * mask).sum(dim=1) / mask.sum(dim=1)
    return log_likelihood.squeeze(-1)

def get_log_prob(text, args, base_tokenizer, base_model):
    tokenized = base_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(args.DEVICE)
    labels = tokenized.input_ids[:, 1:]
    with torch.no_grad():
        logits_score = base_model(**tokenized).logits[:, :-1]
        return get_likelihood(logits_score, labels, base_tokenizer.pad_token_id)

def get_log_probs(texts, args, base_tokenizer, base_model):
    batch_size = args.batch_size
    batch_lprobs = []
    
    if not texts:
        return torch.tensor([])

    for batch in range(0, len(texts), batch_size):
        batch_texts = texts[batch:batch + batch_size]
        tokenized = base_tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(args.DEVICE)
        labels = tokenized.input_ids[:, 1:]
        with torch.no_grad():
            logits_score = base_model(**tokenized).logits[:, :-1]
            lprobs = get_likelihood(logits_score, labels, base_tokenizer.pad_token_id)
            batch_lprobs.append(lprobs)
            
    if not batch_lprobs:
        return torch.tensor([])
    return torch.cat(batch_lprobs, dim=0)


def _sample_from_model(texts, args, tokenizer, model, truncate_ratio=0.5):

    truncated_texts = []
    target_word_counts = []
    
    for t in texts:
        words = t.split(' ')
        n_words = len(words)
        target_word_counts.append(n_words)
        # 截断一半
        truncated_text = ' '.join(words[: int(n_words * truncate_ratio)])
        truncated_texts.append(truncated_text)

    all_encoded = tokenizer(truncated_texts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(args.DEVICE)

    sampling_kwargs = {'temperature': args.temperature}
    if args.do_top_p:
        sampling_kwargs['top_p'] = args.top_p
    elif args.do_top_k:
        sampling_kwargs['top_k'] = args.top_k

    with torch.no_grad():
        outputs = model.generate(
            **all_encoded, 
            min_length=150, 
            max_length=1024, 
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id, # 显式使用 pad_token_id
            eos_token_id=tokenizer.eos_token_id,
            **sampling_kwargs
        )
    
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    final_outputs = []
    for i, text in enumerate(decoded):
        text = clean_text(text)
        text = truncate_text_to_sentences(text, min_word_count=target_word_counts[i])
        final_outputs.append(text)

    return final_outputs

def get_regen_samples(text, args, tokenizer, model):
    texts = [text] * args.regen_number
    sampled_texts = []
    batch_size = args.batch_size
    
    for i in range(0, len(texts), batch_size):
        batch_input = texts[i : i + batch_size]
        batch_output = _sample_from_model(batch_input, args, tokenizer, model, truncate_ratio=args.truncate_ratio)
        sampled_texts.extend(batch_output)
        
    return sampled_texts

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
    # 1. 加载模型与 Tokenizer
    logging.info(f"Loading base model: {args.base_model}...")
    base_tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    
    if base_tokenizer.pad_token is None:
        base_tokenizer.pad_token = base_tokenizer.eos_token
        logging.info("Set base_tokenizer.pad_token to eos_token")
    # --------------------------------

    base_model = AutoModelForCausalLM.from_pretrained(args.base_model).eval().to(args.DEVICE)

    if args.regen_model == args.base_model:
        logging.info("Regen model is same as Base model. Sharing memory.")
        regen_tokenizer = base_tokenizer
        regen_model = base_model
    else:
        logging.info(f"Loading regen model: {args.regen_model}...")
        regen_tokenizer = AutoTokenizer.from_pretrained(args.regen_model)
        
        if regen_tokenizer.pad_token is None:
            regen_tokenizer.pad_token = regen_tokenizer.eos_token
            logging.info("Set regen_tokenizer.pad_token to eos_token")
        # ---------------------------------------
        
        regen_model = AutoModelForCausalLM.from_pretrained(args.regen_model).eval().to(args.DEVICE)

    filenames = args.test_data_path.split(",")
    for filename in filenames:
        logging.info(f"Processing file: {filename}")
        
        test_data = load_data_flexible(filename)
        
        output_stream = filename.split(".json")[0] + "_dna_gpt_stream.jsonl"
        processed_set = get_processed_indices(output_stream)
        logging.info(f"Resuming: {len(processed_set)} items already processed.")

        fout = open(output_stream, "a", encoding="utf-8")

        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        for item in tqdm.tqdm(test_data, desc="DNA-GPT Pipeline"):
            if item["text"] in processed_set:
                continue

            text = item["text"]
            
            if "regen_text" not in item:
                regens = get_regen_samples(text, args, regen_tokenizer, regen_model)
                item["regen_text"] = regens
            else:
                regens = item["regen_text"]

            lprob = get_log_prob(text, args, base_tokenizer, base_model)
            lprob_regens = get_log_probs(regens, args, base_tokenizer, base_model)
            
            wscore = lprob[0] - lprob_regens.mean()
            dna_gpt_score = wscore.item()

            record = {
                "text": text,
                "label": item["label"],
                "regen_text": regens,
                "dna_gpt_score": dna_gpt_score
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
                    if data["label"] == "human":
                        human_scores.append(data["dna_gpt_score"])
                    elif data["label"] == "llm":
                        llm_scores.append(data["dna_gpt_score"])
                except:
                    continue

        human_scores = [i for i in human_scores if np.isfinite(i)]
        llm_scores = [i for i in llm_scores if np.isfinite(i)]

        roc_auc, optimal_threshold, conf_matrix, precision, recall, f1, accuracy, *_ = get_roc_metrics(
            human_scores, llm_scores
        )

        correctness_sequence = []
        for item in final_data:
            score = item["dna_gpt_score"]
            label = item["label"]
            pred_label = "llm" if score >= optimal_threshold else "human"
            is_correct = 1 if pred_label == label else 0
            correctness_sequence.append(is_correct)
            item["prediction"] = pred_label
            item["is_correct"] = bool(is_correct)

        seq_as_str = "".join(map(str, correctness_sequence))
        logging.info(f"ROC AUC: {roc_auc:.4f} | Threshold: {optimal_threshold:.4f}")

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

        data_out = filename.split(".json")[0] + "_dna_gpt_data.json"
        with open(data_out, "w", encoding="utf-8") as f:
            json.dump(final_data, f, indent=4, ensure_ascii=False)

        res_out = filename.split(".json")[0] + "_dna_gpt_result.json"
        with open(res_out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
            
        logging.info(f"Done. Results saved to {res_out}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data_path', type=str, required=True)
    parser.add_argument('--base_model', default="EleutherAI/gpt-neo-2.7B", type=str)
    parser.add_argument('--regen_model', default="t5-large", type=str)
    parser.add_argument('--DEVICE', default="cuda", type=str)
    parser.add_argument('--batch_size', default=5, type=int)
    parser.add_argument('--truncate_ratio', default=0.5, type=float)
    parser.add_argument('--regen_number', default=10, type=int)
    parser.add_argument('--top_k', type=int, default=40)
    parser.add_argument('--do_top_p', action='store_true', default=True)
    parser.add_argument('--do_top_k', action='store_true', default=False)
    parser.add_argument('--top_p', type=float, default=0.96)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--seed', default=2026, type=int)
    args = parser.parse_args()

    if not hasattr(args, "DEVICE"):
        args.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    experiment(args)