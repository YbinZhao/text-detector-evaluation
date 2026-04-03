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


# =================================================
# 0. Config
# =================================================
CSV_PATH = r"D:\py\designirt\all_results_item_format.csv"
EXP_DIR = "myexperiment/14-kfold"

DATASET_JSONL = os.path.join(EXP_DIR, "irt_train.jsonl")
MODEL_PREFIX = os.path.join(EXP_DIR, "irt_model_") + os.sep

# Candidate dimensions to evaluate.
CANDIDATE_DIMS = [2, 3, 4, 5]
LR = 0.01
EPOCHS = 2000
DEVICE = "cuda"  # or "cpu"

# K-fold CV settings.
NUM_FOLDS = 5
SHUFFLE_FOLDS = True
RANDOM_SEED = 42


def compute_validation_bce(responses_all, val_indices, A, B):
    """
    Compute BCE by estimating theta from seen items and predicting unseen items.
    """
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


def build_kfold_splits(n_samples, num_folds, shuffle=True, random_seed=42):
    """
    Return a list of (train_idx, val_idx) for K-fold cross-validation.
    """
    k = min(num_folds, n_samples)
    assert k >= 2, "Need at least 2 samples for K-fold CV."

    indices = np.arange(n_samples)
    if shuffle:
        rng = np.random.default_rng(random_seed)
        indices = rng.permutation(indices)

    folds = np.array_split(indices, k)
    splits = []
    for i in range(k):
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        splits.append((train_idx.tolist(), val_idx.tolist()))
    return splits


os.makedirs(EXP_DIR, exist_ok=True)

print("[1] Loading CSV...")
df = pd.read_csv(CSV_PATH)
model_names = df.iloc[:, 0].astype(str).values
responses = df.iloc[:, 1:].values.astype(int)

print("responses shape:", responses.shape)
assert set(np.unique(responses)).issubset({0, 1}), "responses must contain only 0/1."

print("[2] Creating full-data IRT dataset...")
create_irt_dataset(responses, DATASET_JSONL)
assert os.path.exists(DATASET_JSONL)

print("[3] Selecting best dimension with K-fold CV...")
assert len(CANDIDATE_DIMS) > 0, "CANDIDATE_DIMS must not be empty."

splits = build_kfold_splits(
    n_samples=responses.shape[0],
    num_folds=NUM_FOLDS,
    shuffle=SHUFFLE_FOLDS,
    random_seed=RANDOM_SEED,
)
print(f"using {len(splits)} folds")

dim_results = []
for d in CANDIDATE_DIMS:
    fold_losses = []
    print(f"  - Evaluating D={d} ...")

    for fold_i, (train_idx, val_idx) in enumerate(splits, start=1):
        fold_dataset = os.path.join(EXP_DIR, f"irt_val_train_fold{fold_i}.jsonl")
        create_irt_dataset(responses[train_idx], fold_dataset)

        fold_model_prefix = os.path.join(EXP_DIR, f"irt_val_model_d{d}_fold{fold_i}_") + os.sep
        os.makedirs(fold_model_prefix, exist_ok=True)

        train_irt_model(
            dataset_name=fold_dataset,
            model_name=fold_model_prefix,
            D=d,
            lr=LR,
            epochs=EPOCHS,
            device=DEVICE,
        )

        A_d, B_d, _ = load_irt_parameters(fold_model_prefix)
        fold_bce = compute_validation_bce(responses, val_idx, A_d, B_d)
        fold_losses.append(fold_bce)
        print(f"    fold {fold_i}/{len(splits)}: val_bce={fold_bce:.6f}")

    cv_bce = float(np.mean(fold_losses))
    dim_results.append({"D": d, "cv_bce": cv_bce, "fold_bce": str([round(x, 6) for x in fold_losses])})
    print(f"    D={d}, cv_bce={cv_bce:.6f}")

best_row = min(dim_results, key=lambda x: x["cv_bce"])
best_d = int(best_row["D"])
print(f"[3] Best D selected: {best_d} (cv_bce={best_row['cv_bce']:.6f})")

cv_report = os.path.join(EXP_DIR, "dimension_selection_kfold.csv")
pd.DataFrame(dim_results).to_csv(cv_report, index=False)
print("k-fold dimension report saved to:", cv_report)


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

theta_out = os.path.join(EXP_DIR, f"theta_train_{best_d}d_kfold.csv")
theta_df.to_csv(theta_out, index=False)

print("\n[Done]")
print("best D:", best_d)
print("theta saved to:", theta_out)
print("All outputs saved under:", EXP_DIR)

