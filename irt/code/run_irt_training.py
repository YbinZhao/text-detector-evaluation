import os
import pandas as pd
import numpy as np

from irt import (
    create_irt_dataset,
    train_irt_model,
    load_irt_parameters
)


CSV_PATH = r""  # 作答矩阵 CSV
EXP_DIR = ""  # 实验输出目录

DATASET_JSONL = os.path.join(EXP_DIR, "irt_train.jsonl")

# ⚠️ 核心：MODEL_PREFIX 必须以 os.sep 结尾
MODEL_PREFIX = os.path.join(EXP_DIR, "irt_model_") + os.sep

# ===== IRT 超参数 =====
D = 1
LR = 0.01
EPOCHS = 2000
DEVICE = "cuda"  # 或 "cuda"


os.makedirs(EXP_DIR, exist_ok=True)


print("[1] Loading CSV...")
df = pd.read_csv(CSV_PATH)

# 默认：第一列是模型名，其余列是 0/1
model_names = df.iloc[:, 0].astype(str).values
responses = df.iloc[:, 1:].values.astype(int)

print("responses shape:", responses.shape)  # (n_models, n_items)

# 基本合法性检查
assert set(np.unique(responses)).issubset({0, 1}), "responses 必须只包含 0/1"


print("[2] Creating IRT dataset...")
create_irt_dataset(responses, DATASET_JSONL)
assert os.path.exists(DATASET_JSONL)


print("[3] Training IRT model (D=5)...")
train_irt_model(
    dataset_name=DATASET_JSONL,
    model_name=MODEL_PREFIX,  # ⚠️ 一定要是“以分隔符结尾的前缀”
    D=D,
    lr=LR,
    epochs=EPOCHS,
    device=DEVICE
)


print("[4] Loading IRT parameters...")
A, B, Theta = load_irt_parameters(MODEL_PREFIX)

print("A (discrimination) shape:", np.shape(A))
print("B (difficulty) shape:", np.shape(B))
print("Theta (ability) raw shape:", np.shape(Theta))


Theta = np.asarray(Theta, dtype=float)

if Theta.ndim == 3 and Theta.shape[-1] == 1:
    Theta = np.squeeze(Theta, axis=-1)

if Theta.ndim == 1:
    Theta = Theta[:, None]

assert Theta.ndim == 2, f"Theta must be 2D, got {Theta.shape}"
assert Theta.shape[1] == D, f"Theta second dim must be D={D}, got {Theta.shape}"

print("Theta (ability) fixed shape:", Theta.shape)

theta_df = pd.DataFrame(
    Theta,
    columns=[f"theta_{i+1}" for i in range(D)]
)
theta_df.insert(0, "model", model_names)

theta_out = os.path.join(EXP_DIR, "theta_train_5d.csv")
theta_df.to_csv(theta_out, index=False)

print("\n[Done]")
print("theta saved to:", theta_out)
print("All outputs saved under:", EXP_DIR)
