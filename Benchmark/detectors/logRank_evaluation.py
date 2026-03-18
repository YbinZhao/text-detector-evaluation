# -*- coding: utf-8 -*-
import logging
import random
import torch
import tqdm
import argparse
import json
import os
import sys
import numpy as np
from rank import get_rank
from metrics import get_roc_metrics
from transformers import AutoTokenizer, AutoModelForCausalLM

# 配置日志输出
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

# ===================== 数据集加载兼容逻辑 =====================
def load_data_flexible(path):
    logging.info(f"正在加载数据集: {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    
    formatted_data = []
    # 兼容处理：如果是字典格式 {"original": [], "sampled": []}
    if isinstance(raw_data, dict) and "original" in raw_data:
        logging.info("检测到字典格式数据集，正在进行平铺转换...")
        for text in raw_data["original"]:
            formatted_data.append({"text": text, "label": "human"})
        for text in raw_data["sampled"]:
            formatted_data.append({"text": text, "label": "llm"})
    # 兼容处理：如果是列表格式
    elif isinstance(raw_data, list):
        formatted_data = raw_data
    else:
        raise ValueError("未知的数据格式。请提供列表或带有 'original'/'sampled' 键的字典。")
    
    return formatted_data

# ===================== 主实验逻辑 =====================
def experiment(args):
    # 1. 加载模型与分词器
    logging.info(f"正在加载模型: {args.base_model}...")
    base_tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    
    # 修复 GPT-2/GPT-Neo 等模型缺失 pad_token 的问题
    if base_tokenizer.pad_token is None:
        base_tokenizer.pad_token = base_tokenizer.eos_token
        
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model)
    base_model.eval().to(args.DEVICE)

    filenames = args.test_data_path.split(",")
    for filename in filenames:
        logging.info(f"开始处理文件: {filename}")
        test_data = load_data_flexible(filename)

        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        predictions = {'human': [], 'llm': []}
        
        # 2. 计算 LogRank 分数
        # 逻辑：机器生成的文本通常 Rank 较小，取负数后分数越高越像 AI
        for item in tqdm.tqdm(test_data, desc="计算 LogRank"):
            text = item["text"]
            label = item["label"]

            # 调用 get_rank 计算平均排名，log=True 使用 LogRank
            rank_val = get_rank(text, args, base_tokenizer, base_model, log=True)
            # 处理可能的异常返回
            score = -rank_val if rank_val is not None else -1e9
            item["text_logrank"] = score

            if label == "human":
                predictions['human'].append(score)
            elif label == "llm":
                predictions['llm'].append(score)
            else:
                raise ValueError(f"未知标签: {label}")

        # 过滤无效数据
        predictions['human'] = [i for i in predictions['human'] if np.isfinite(i)]
        predictions['llm'] = [i for i in predictions['llm'] if np.isfinite(i)]

        # 3. 计算指标并获取最优阈值
        # 假设 get_roc_metrics 返回 7 个值 (roc_auc 到 accuracy)
        res_metrics = get_roc_metrics(predictions['human'], predictions['llm'])
        roc_auc, optimal_threshold, conf_matrix, precision, recall, f1, accuracy = res_metrics[:7]

        # 4. 生成 0/1 正确性序列
        # 1 代表预测正确，0 代表预测错误
        correctness_sequence = []
        for item in test_data:
            score = item["text_logrank"]
            # 预测：分数 >= 阈值判定为 llm，否则为 human
            pred_label = "llm" if score >= optimal_threshold else "human"
            is_correct = 1 if pred_label == item["label"] else 0
            
            correctness_sequence.append(is_correct)
            item["prediction"] = pred_label
            item["is_correct"] = bool(is_correct)

        seq_as_str = "".join(map(str, correctness_sequence))

        # 5. 整合结果并保存
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

        logging.info(f"结果预览: AUC={roc_auc:.4f}, Accuracy={accuracy:.4f}")

        # 保存详细预测数据
        out_data_path = filename.split(".json")[0] + "_logRank_data.json"
        with open(out_data_path, "w", encoding="utf-8") as f:
            json.dump(test_data, f, indent=4, ensure_ascii=False)

        # 保存统计指标结果
        out_res_path = filename.split(".json")[0] + "_logRank_result.json"
        with open(out_res_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
            
        logging.info(f"保存完毕。")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data_path', type=str, required=True,
                        help="测试数据路径，支持逗号分隔多个文件。")
    parser.add_argument('--base_model', default="EleutherAI/gpt-neo-2.7B", type=str)
    parser.add_argument('--DEVICE', default="cuda", type=str)
    parser.add_argument('--seed', default=2023, type=int)
    args = parser.parse_args()

    # 自动检查设备可用性
    if args.DEVICE == "cuda" and not torch.cuda.is_available():
        logging.warning("指定了 CUDA 但不可用，已切换至 CPU。")
        args.DEVICE = "cpu"

    experiment(args)