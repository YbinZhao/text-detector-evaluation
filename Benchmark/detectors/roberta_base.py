# -*- coding: utf-8 -*-
import logging
import numpy as np
import torch
import tqdm
import argparse
import json
import sys
import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import precision_recall_fscore_support

try:
    from metrics import get_roc_metrics
except ImportError:
    logging.error("无法找到 metrics.py，请确保该文件在当前目录下。")

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

def load_data_local(path):

    logging.info(f"正在读取文件: {path}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "original" not in data or "sampled" not in data:
            raise ValueError("数据集必须包含 'original' 和 'sampled' 键。")
        return data
    except Exception as e:
        logging.error(f"加载数据失败: {e}")
        sys.exit(1)

def experiment(args):

    logging.info(f"正在加载监督模型: {args.base_model}...")
    try:
        detector = AutoModelForSequenceClassification.from_pretrained(args.base_model).to(args.DEVICE)
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        detector.eval()
        logging.info("模型加载完成。")
    except Exception as e:
        logging.error(f"模型加载失败: {e}")
        return

    data = load_data_local(args.test_data_path)
    
    evaluation_queue = []
    for txt in data["original"]:
        evaluation_queue.append({"text": txt, "label": "human", "true_id": 1}) # 1: Real
    for txt in data["sampled"]:
        evaluation_queue.append({"text": txt, "label": "llm", "true_id": 0})    # 0: Fake

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    test_data = []
    all_labels = []
    all_preds = []
    correctness_sequence = []
    
    logging.info(f"开始逐条评估，共 {len(evaluation_queue)} 条文本...")
    
    for item in tqdm.tqdm(evaluation_queue, desc=f"Evaluating {args.base_model}"):
        text = item["text"]
        true_id = item["true_id"]
        
        tokenized = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt").to(args.DEVICE)
        
        with torch.no_grad():
            outputs = detector(**tokenized)
            logits = outputs.logits
            
            pred_id = logits.argmax(-1).item()
            fake_prob = logits.softmax(-1)[0, 0].item()
        
        all_labels.append(true_id)
        all_preds.append(pred_id)
        
        is_correct = 1 if pred_id == true_id else 0
        correctness_sequence.append(is_correct)
        
        test_data.append({
            "text": text,
            "label": item["label"],
            "text_ll": fake_prob, 
            "prediction": "human" if pred_id == 1 else "llm",
            "is_correct": bool(is_correct)
        })

    predictions_scores = {
        'human': [x["text_ll"] for x in test_data if x["label"] == "human"],
        'llm': [x["text_ll"] for x in test_data if x["label"] == "llm"]
    }
    roc_auc = 0.0
    try:
        res_metrics = get_roc_metrics(predictions_scores['human'], predictions_scores['llm'])
        roc_auc = res_metrics[0]
    except Exception:
        pass

    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', pos_label=1)

    seq_as_str = "".join(map(str, correctness_sequence))
    
    result_summary = {
        "roc_auc": roc_auc,
        "accuracy": np.mean(correctness_sequence),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "correctness_sequence": seq_as_str
    }

    input_path = args.test_data_path
    base_name = input_path[:-5] if input_path.endswith(".json") else input_path
    data_out = f'{base_name}_roberta-base_data.json'
    res_out = f'{base_name}_roberta-base_result.json'

    with open(data_out, "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=4, ensure_ascii=False)
        
    with open(res_out, "w", encoding="utf-8") as f:
        json.dump(result_summary, f, indent=4, ensure_ascii=False)
        
    logging.info("-" * 30)
    logging.info(f"评估完成！统计指标如下:")
    logging.info(f"Accuracy : {result_summary['accuracy']:.4f}")
    logging.info(f"Precision: {precision:.4f}")
    logging.info(f"Recall   : {recall:.4f}")
    logging.info(f"F1 Score : {f1:.4f}")
    logging.info(f"ROC AUC  : {roc_auc:.4f}")
    logging.info("-" * 30)
    logging.info(f"详细数据保存至: {data_out}")
    logging.info(f"统计结果保存至: {res_out}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data_path', type=str, default="./exp_test/data/xsum_gpt2", help="输入数据集路径")
    parser.add_argument('--base_model', type=str, default="roberta-base-openai-detector", help="模型名称或路径")
    parser.add_argument('--DEVICE', type=str, default="cuda", help="计算设备")
    parser.add_argument('--seed', type=int, default=2026)
    
    args = parser.parse_args()

    if args.DEVICE == "cuda" and not torch.cuda.is_available():
        args.DEVICE = "cpu"

    experiment(args)