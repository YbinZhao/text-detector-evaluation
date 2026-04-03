import os
import pandas as pd
import numpy as np

from irt import (
    create_irt_dataset,
    train_irt_model,
    load_irt_parameters,
    estimate_ability_parameters,
)
from utils import item_curve


CSV_PATH = r"D:\py\designirt\all_results_item_format.csv" 
EXP_DIR = "myexperiment/14-2.1"  

DATASET_JSONL = os.path.join(EXP_DIR, "irt_train.jsonl")


MODEL_PREFIX = os.path.join(EXP_DIR, "irt_model_") + os.sep


CANDIDATE_DIMS = [2, 3, 4, 5, 10, 15]
LR = 0.01
EPOCHS = 2000
DEVICE = "cuda"  # 或 "cpu"


def compute_validation_bce(responses_all, val_indices, A, B):

    seen_items = list(range(0, responses_all.shape[1], 2))
    unseen_items = list(range(1, responses_all.shape[1], 2))
    assert len(seen_items) > 0 and len(unseen_items) > 0, "Need both seen and unseen items."

    eps = 1e-10
    losses = []

    for idx in val_indices:
        theta = estimate_ability_parameters(
            responses_all[idx][seen_items],
            A[:, :, seen_items],
            B[:, :, seen_items],
        )
        pred = item_curve(theta, A[:, :, unseen_items], B[:, :, unseen_items]).reshape(-1)
        true = responses_all[idx][unseen_items].astype(float)
        bce = -np.mean(true * np.log(pred + eps) + (1.0 - true) * np.log(1.0 - pred + eps))
        losses.append(float(bce))

    return float(np.mean(losses))


os.makedirs(EXP_DIR, exist_ok=True)

print("[1] Loading CSV...")
df = pd.read_csv(CSV_PATH)

model_names = df.iloc[:, 0].astype(str).values
responses = df.iloc[:, 1:].values.astype(int)

print("responses shape:", responses.shape)  # (n_models, n_items)
assert set(np.unique(responses)).issubset({0, 1}), "responses 必须只包含 0/1"

print("[2] Creating full-data IRT dataset...")
create_irt_dataset(responses, DATASET_JSONL)
assert os.path.exists(DATASET_JSONL)

print("[3] Selecting best dimension with validation...")
assert len(CANDIDATE_DIMS) > 0, "CANDIDATE_DIMS must not be empty."

val_ind = list(range(0, responses.shape[0], 5))
train_ind = [i for i in range(responses.shape[0]) if i not in val_ind]
assert len(val_ind) > 0 and len(train_ind) > 0, "Need non-empty train/val split."

val_dataset_jsonl = os.path.join(EXP_DIR, "irt_val_train.jsonl")
create_irt_dataset(responses[train_ind], val_dataset_jsonl)

dim_losses = []
for d in CANDIDATE_DIMS:
    val_model_prefix = os.path.join(EXP_DIR, f"irt_val_model_d{d}_") + os.sep
    os.makedirs(val_model_prefix, exist_ok=True)

    print(f"  - Training validation model with D={d} ...")
    train_irt_model(
        dataset_name=val_dataset_jsonl,
        model_name=val_model_prefix,
        D=d,
        lr=LR,
        epochs=EPOCHS,
        device=DEVICE,
    )

    A_d, B_d, _ = load_irt_parameters(val_model_prefix)
    val_bce = compute_validation_bce(responses, val_ind, A_d, B_d)
    dim_losses.append((d, val_bce))
    print(f"    D={d}, val_bce={val_bce:.6f}")

best_d, best_loss = min(dim_losses, key=lambda x: x[1])
print(f"[3] Best D selected: {best_d} (val_bce={best_loss:.6f})")

dim_report = os.path.join(EXP_DIR, "dimension_selection.csv")
pd.DataFrame(dim_losses, columns=["D", "val_bce"]).to_csv(dim_report, index=False)
print("dimension report saved to:", dim_report)

print(f"[4] Training final IRT model with D={best_d} on full data...")
train_irt_model(
    dataset_name=DATASET_JSONL,
    model_name=MODEL_PREFIX,
    D=best_d,
    lr=LR,
    epochs=EPOCHS,
    device=DEVICE,
)

print("[5] Loading final IRT parameters...")
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
assert Theta.shape[1] == best_d, f"Theta second dim must be D={best_d}, got {Theta.shape}"

print("Theta (ability) fixed shape:", Theta.shape)

theta_df = pd.DataFrame(Theta, columns=[f"theta_{i+1}" for i in range(best_d)])
theta_df.insert(0, "model", model_names)

theta_out = os.path.join(EXP_DIR, f"theta_train_{best_d}d.csv")
theta_df.to_csv(theta_out, index=False)

print("\n[Done]")
print("best D:", best_d)
print("theta saved to:", theta_out)
print("All outputs saved under:", EXP_DIR)

