# -*- coding: utf-8 -*-

import logging
import random
import numpy as np
import torch
import tqdm
import argparse
import json
import os
from metrics import get_roc_metrics
from transformers import AutoTokenizer, AutoModelForCausalLM

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ============================
# LRR 核心计算函数
# ============================
def get_rank(text, args, tokenizer, model, log=False):
    with torch.no_grad():
        if text == "":
            return None
        else:
            tokenized = tokenizer(text, return_tensors="pt").to(args.DEVICE)
            logits = model(**tokenized).logits[:, :-1]
            labels = tokenized.input_ids[:, 1:]

            # 获取 Rank
            matches = (logits.argsort(-1, descending=True) == labels.unsqueeze(-1)).nonzero()

            assert matches.shape[1] == 3, f"Expected 3 dimensions in matches tensor, got {matches.shape}"

            ranks, timesteps = matches[:, -1], matches[:, -2]

            assert (timesteps == torch.arange(len(timesteps)).to(
                timesteps.device)).all(), "Expected one match per timestep"

            ranks = ranks.float() + 1
            if log:
                ranks = torch.log(ranks)

            return ranks.float().mean().item()

def get_ll(text, args, tokenizer, model):
    with torch.no_grad():
        tokenized = tokenizer(text, return_tensors="pt").to(args.DEVICE)
        labels = tokenized['input_ids']
        if labels.nelement() == 0:
            logging.error(f"Empty input: {text}")
            return 0
        else:
            return -model(**tokenized, labels=labels).loss.item()

# ============================
# 主流程
# ============================
def experiment(args):
    # 1. 加载模型
    logging.info(f"Loading base model: {args.base_model}...")
    base_tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model)
    base_model.eval()
    base_model.to(args.DEVICE)

    filenames = args.test_data_path.split(",")
    for filename in filenames:
        logging.info(f"Processing: {filename}")
        raw_data = json.load(open(filename, "r"))

        # 2. 数据集兼容性处理 (Dict -> List)
        test_data = []
        if isinstance(raw_data, dict) and "original" in raw_data:
            for text in raw_data["original"]:
                test_data.append({"text": text, "label": "human"})
            for text in raw_data["sampled"]:
                test_data.append({"text": text, "label": "llm"})
        else:
            test_data = raw_data # 假设已经是列表格式

        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        predictions = {'human': [], 'llm': []}
        
        # 3. 计算分数循环
        for item in tqdm.tqdm(test_data, desc="Calculating LRR"):
            text = item["text"]
            label = item["label"]

            # 计算 Likelihood 和 LogRank
            ll = get_ll(text, args, base_tokenizer, base_model)
            logrank = get_rank(text, args, base_tokenizer, base_model, log=True)
            
            # LRR 公式: -LL / LogRank
            # 注意: 如果 logrank 为 0 (极罕见情况), 需要处理除零错误
            lrr_score = -ll / logrank if logrank != 0 else 0
            
            item["text_ll"] = ll
            item["text_logrank"] = logrank
            item["text_LRR"] = lrr_score

            if label == "human":
                predictions['human'].append(lrr_score)
            elif label == "llm":
                predictions['llm'].append(lrr_score)
            else:
                raise ValueError(f"Unknown label {label}")

        # 过滤无效值
        predictions['human'] = [i for i in predictions['human'] if np.isfinite(i)]
        predictions['llm'] = [i for i in predictions['llm'] if np.isfinite(i)]

        # 4. 计算 ROC 指标 (获取最佳阈值)
        # 注意：这里假设 get_roc_metrics 返回 7 个值，如果你的 metrics.py 有变化，请相应调整
        roc_auc, optimal_threshold, conf_matrix, precision, recall, f1, accuracy, *_ = get_roc_metrics(
            predictions['human'], predictions['llm'])

        # 5. 生成 0/1 正确性序列
        correctness_sequence = []
        for item in test_data:
            score = item["text_LRR"]
            label = item["label"]
            
            # LRR 逻辑：通常机器生成的 LRR 值比人类的高
            # Prediction
            pred_label = "llm" if score >= optimal_threshold else "human"
            
            # Check correctness
            is_correct = 1 if pred_label == label else 0
            correctness_sequence.append(is_correct)

            # 保存回 item
            item["prediction"] = pred_label
            item["is_correct"] = bool(is_correct)

        seq_as_str = "".join(map(str, correctness_sequence))
        print(f"\nCorrectness Sequence: {seq_as_str}")
        print(f"ROC AUC: {roc_auc:.4f} | Optimal Threshold: {optimal_threshold:.4f}")

        # 6. 构造结果字典
        result = {
            "roc_auc": roc_auc,
            "optimal_threshold": optimal_threshold,
            "conf_matrix": conf_matrix,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
            "correctness_sequence": seq_as_str  # 新增字段
        }

        # 7. 保存文件
        # 保存详细数据
        data_out = filename.split(".json")[0] + "_LRR_data.json"
        with open(data_out, "w", encoding="utf-8") as f:
            json.dump(test_data, f, indent=4, ensure_ascii=False)

        # 保存统计结果
        res_out = filename.split(".json")[0] + "_LRR_result.json"
        with open(res_out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
            
        logging.info(f"Done. Saved to {res_out}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data_path', type=str, required=True,
                        help="Path to test data (json file).")
    parser.add_argument('--base_model', default="EleutherAI/gpt-neo-2.7B", type=str)
    parser.add_argument('--DEVICE', default="cuda", type=str, required=False)
    parser.add_argument('--seed', default=2026, type=int, required=False)
    args = parser.parse_args()

    # 设备检查
    if not hasattr(args, "DEVICE"):
         args.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    experiment(args)