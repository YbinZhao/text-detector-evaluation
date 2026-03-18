# -*- coding: utf-8 -*-
import logging
import random
import torch
import tqdm
import argparse
import json
import numpy as np
import os
import sys

from binoculars_detector import Binoculars
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix,
    precision_score, recall_score,
    accuracy_score, f1_score
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

# ===================== 工具函数 =====================
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_processed_texts(out_path):
    """读取已生成的流式文件，返回已处理文本的集合，用于去重续传"""
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

# ===================== 指标计算 =====================
def get_roc_metrics(real_preds, sample_preds):
    """
    real_preds: Human scores (Label 0)
    sample_preds: LLM scores (Label 1)
    """
    real_labels = [0] * len(real_preds) + [1] * len(sample_preds)
    predicted_probs = real_preds + sample_preds

    # 转换为 array
    real_labels = np.array(real_labels)
    predicted_probs = np.array(predicted_probs)

    fpr, tpr, thresholds = roc_curve(real_labels, predicted_probs)
    roc_auc = auc(fpr, tpr)

    # Youden's J statistic (TPR - FPR) 寻找最佳阈值
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    predictions = np.array([1 if p >= optimal_threshold else 0 for p in predicted_probs])
    
    conf_matrix = confusion_matrix(real_labels, predictions)
    precision = precision_score(real_labels, predictions, zero_division=0)
    recall = recall_score(real_labels, predictions, zero_division=0)
    f1 = f1_score(real_labels, predictions, zero_division=0)
    accuracy = accuracy_score(real_labels, predictions)

    # TPR @ FPR = 0.01%
    tpr_at_fpr_0_01 = np.interp(0.0001, fpr, tpr)

    return (
        float(roc_auc),
        float(optimal_threshold),
        conf_matrix.tolist(),
        float(precision),
        float(recall),
        float(f1),
        float(accuracy),
        float(tpr_at_fpr_0_01),
    )

# ===================== 主实验逻辑 =====================
def experiment(args):
    # 初始化 Binoculars
    # 注意：Binoculars 默认会自动使用 CUDA，如果显存紧张可能需要额外配置
    logging.info("Initializing Binoculars model...")
    bino = Binoculars(
        mode="accuracy",
        max_token_observed=args.tokens_seen
    )

    filenames = args.test_data_path.split(",")

    for filename in filenames:
        logging.info(f"Processing: {filename}")
        raw_data = load_json(filename)

        # 1. 数据集兼容性处理 (Dict -> List)
        test_data = []
        if isinstance(raw_data, dict) and "original" in raw_data:
            for text in raw_data["original"]:
                test_data.append({"text": text, "label": "human"})
            for text in raw_data["sampled"]:
                test_data.append({"text": text, "label": "llm"})
        else:
            test_data = raw_data

        # 2. 准备流式输出文件 (Streaming Output)
        output_stream_path = filename.split(".json")[0] + "_bino_stream.jsonl"
        processed_set = get_processed_texts(output_stream_path)
        logging.info(f"Resuming: {len(processed_set)} items already processed.")

        # 以追加模式打开
        fout = open(output_stream_path, "a", encoding="utf-8")

        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        # 3. 实时计算循环
        for item in tqdm.tqdm(test_data, desc="Binoculars Scoring"):
            text = item["text"]
            
            # 断点续传：跳过已处理
            if text in processed_set:
                continue

            # Binoculars 计算
            # 注意：Binoculars 原始分数越低越像 AI (Perplexity gap)
            # 这里取负号 (-score)，使得 分数越高越像 AI，符合 ROC 计算习惯 (Label 1 = LLM)
            raw_score = bino.compute_score(text)
            score = -raw_score 
            
            record = {
                "text": text,
                "label": item["label"],
                "bino_score": score
            }

            # 实时写入磁盘
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()
        
        fout.close()

        # 4. 后处理：读取流文件，计算指标和 0/1 序列
        logging.info("Calculating metrics from stream file...")
        
        final_data = []
        human_scores = []
        llm_scores = []

        with open(output_stream_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    final_data.append(data)
                    if data["label"] == "human":
                        human_scores.append(data["bino_score"])
                    elif data["label"] == "llm":
                        llm_scores.append(data["bino_score"])
                except:
                    continue

        # 计算 ROC
        (
            roc_auc,
            optimal_threshold,
            conf_matrix,
            precision,
            recall,
            f1,
            accuracy,
            tpr_at_fpr_0_01,
        ) = get_roc_metrics(human_scores, llm_scores)

        # 生成 0/1 正确性序列
        correctness_sequence = []
        for item in final_data:
            score = item["bino_score"]
            label = item["label"]
            
            # 根据最佳阈值判定
            pred_label = "llm" if score >= optimal_threshold else "human"
            is_correct = 1 if pred_label == label else 0
            
            correctness_sequence.append(is_correct)
            
            # 回填到 item
            item["prediction"] = pred_label
            item["is_correct"] = bool(is_correct)

        seq_as_str = "".join(map(str, correctness_sequence))
        logging.info(f"Correctness Sequence Length: {len(seq_as_str)}")
        logging.info(f"ROC AUC: {roc_auc:.4f} | Threshold: {optimal_threshold:.4f}")

        # 5. 保存最终结果
        result = {
            "roc_auc": roc_auc,
            "optimal_threshold": optimal_threshold,
            "conf_matrix": conf_matrix,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
            "tpr_at_fpr_0_01": tpr_at_fpr_0_01,
            "correctness_sequence": seq_as_str
        }

        # 保存详细数据
        data_out = filename.split(".json")[0] + "_bino_data.json"
        with open(data_out, "w", encoding="utf-8") as f:
            json.dump(final_data, f, indent=4, ensure_ascii=False)

        # 保存统计结果
        res_out = filename.split(".json")[0] + "_bino_result.json"
        with open(res_out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
            
        logging.info(f"Done. Results saved to {res_out}")

# ===================== CLI =====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test_data_path",
        type=str,
        required=True,
        help="Comma-separated json files"
    )
    parser.add_argument(
        "--tokens_seen",
        type=int,
        default=512,
        help="Number of tokens seen by Binoculars"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026
    )
    args = parser.parse_args()

    experiment(args)