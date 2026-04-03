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

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

try:
    from metrics import get_roc_metrics
except ImportError:
    logging.error("无法找到 metrics.py，请确保该文件在当前目录下。")

def get_ll(text, args, tokenizer, model):
    if not text or len(text.strip()) == 0:
        return 0.0
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    with torch.no_grad():
        tokenized = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(args.DEVICE)
        labels = tokenized['input_ids']
        if labels.nelement() == 0:
            return 0.0
        return -model(**tokenized, labels=labels).loss.item()

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
    
    logging.info(f"成功加载数据，共计 {len(test_data)} 条。")
    return test_data

def experiment(args):
    logging.info(f"正在准备加载模型: {args.base_model} (这可能需要一些时间，取决于模型大小和网络)...")
    try:
        base_tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        base_model = AutoModelForCausalLM.from_pretrained(args.base_model)
        logging.info(f"模型权重已加载到内存，正在移动到设备: {args.DEVICE}...")
        base_model.eval().to(args.DEVICE)
        logging.info("模型加载完成。")
    except Exception as e:
        logging.error(f"模型加载失败: {e}")
        return

    filenames = args.test_data_path.split(",")
    for filename in filenames:
        test_data = load_data_flexible(filename)

        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        predictions = {'human': [], 'llm': []}
        
        logging.info("开始计算 Likelihood 分数...")
        for item in tqdm.tqdm(test_data, desc="Evaluating"):
            text = item["text"]
            label = item["label"]

            score = get_ll(text, args, base_tokenizer, base_model)
            item["text_ll"] = score

            if label == "human":
                predictions['human'].append(score)
            elif label == "llm":
                predictions['llm'].append(score)
            else:
                raise ValueError(f"Unknown label {label}")

        predictions['human'] = [i for i in predictions['human'] if np.isfinite(i)]
        predictions['llm'] = [i for i in predictions['llm'] if np.isfinite(i)]

        logging.info("正在计算 ROC 指标并寻找最佳阈值...")
        res_metrics = get_roc_metrics(predictions['human'], predictions['llm'])
        roc_auc, optimal_threshold, conf_matrix, precision, recall, f1, accuracy = res_metrics[:7]

        correctness_sequence = []
        for item in test_data:
            score = item["text_ll"]
            label = item["label"]
            
            pred_label = "llm" if score >= optimal_threshold else "human"
            is_correct = 1 if pred_label == label else 0
            
            correctness_sequence.append(is_correct)
            item["prediction"] = pred_label
            item["is_correct"] = bool(is_correct)

        seq_as_str = "".join(map(str, correctness_sequence))
        logging.info(f"正确性序列已生成 (长度: {len(seq_as_str)})")

        result = {
            "roc_auc": roc_auc,
            "optimal_threshold": optimal_threshold,
            "conf_matrix": conf_matrix,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
            "correctness_sequence": seq_as_str  # 保存 01 序列
        }

        logging.info(f"最终结果: {result}")
        
        data_out = filename.split(".json")[0] + "_likelihood_data.json"
        with open(data_out, "w", encoding="utf-8") as f:
            json.dump(test_data, f, indent=4, ensure_ascii=False)

        res_out = filename.split(".json")[0] + "_likelihood_result.json"
        with open(res_out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
        
        logging.info(f"任务完成。详细数据保存至: {data_out}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data_path', type=str, required=True)
    parser.add_argument('--base_model', default="EleutherAI/gpt-neo-2.7B", type=str)
    parser.add_argument('--DEVICE', default="cuda", type=str)
    parser.add_argument('--seed', default=2026, type=int)
    args = parser.parse_args()

    if args.DEVICE == "cuda" and not torch.cuda.is_available():
        logging.warning("检测到指定了 CUDA 但硬件不可用，已自动降级至 CPU。")
        args.DEVICE = "cpu"

    experiment(args)