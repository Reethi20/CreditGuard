
import time
import json
import numpy as np
import pandas as pd

from pathlib import Path
from catboost import CatBoostClassifier

from sklearn.model_selection import (
    StratifiedGroupKFold,
    RandomizedSearchCV,
    cross_val_predict
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
    precision_recall_curve
)

# --------------------------------------------------
# 1. PATHS AND SETTINGS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[3]

DATA_PATH = ROOT / "data/raw/creditcard.csv"
SPLIT_PATH = ROOT / "data/processed/experiment_splits.csv"

OUTPUT_DIR = ROOT / "models/experiments"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULT_PATH = OUTPUT_DIR / "catboost_optimization_results.csv"

RANDOM_STATE = 42
N_SPLITS = 5
N_ITER = 8

# --------------------------------------------------
# 2. LOAD ORIGINAL DATA
# --------------------------------------------------

print("\nLoading original dataset...")

df = pd.read_csv(DATA_PATH)
splits = pd.read_csv(SPLIT_PATH)

if "Class" not in df.columns:
    raise ValueError("Target column 'Class' was not found.")

# Identify split-file columns.
index_candidates = ["index", "row_index", "original_index"]
split_candidates = ["split", "partition", "dataset"]

index_col = next(
    (c for c in index_candidates if c in splits.columns),
    None
)

split_col = next(
    (c for c in split_candidates if c in splits.columns),
    None
)

if index_col is None or split_col is None:
    raise ValueError(
        "Could not identify the index and split columns "
        "in experiment_splits.csv. Check its column names."
    )

train_indices = splits.loc[
    splits[split_col].str.lower() == "train", index_col
].astype(int).to_numpy()

val_indices = splits.loc[
    splits[split_col].str.lower().isin(
        ["validation", "val"]
    ), index_col
].astype(int).to_numpy()

if len(train_indices) == 0 or len(val_indices) == 0:
    raise ValueError(
        "Training or validation indices are empty. "
        "Check the split labels in experiment_splits.csv."
    )

# Use only training and validation rows.
# The test partition is deliberately not loaded into this experiment.

X = df.drop(columns=["Class"])
y = df["Class"]

X_train = X.iloc[train_indices].reset_index(drop=True)
y_train = y.iloc[train_indices].reset_index(drop=True)

X_val = X.iloc[val_indices].reset_index(drop=True)
y_val = y.iloc[val_indices].reset_index(drop=True)

print("Training rows:", len(X_train))
print("Validation rows:", len(X_val))
print("Training fraud cases:", int(y_train.sum()))
print("Validation fraud cases:", int(y_val.sum()))

# --------------------------------------------------
# 3. DUPLICATE GROUPS
# --------------------------------------------------

print("\nCreating duplicate groups...")

groups = pd.util.hash_pandas_object(
    X_train,
    index=False
).to_numpy()

print("Unique training groups:", len(np.unique(groups)))

# --------------------------------------------------
# 4. GROUP-AWARE CROSS-VALIDATION
# --------------------------------------------------

cv = StratifiedGroupKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

cv_splits = list(
    cv.split(X_train, y_train, groups)
)

# Verify no identical feature rows cross between folds.
for fold_number, (train_idx, valid_idx) in enumerate(
    cv_splits, start=1
):
    train_groups = set(groups[train_idx])
    valid_groups = set(groups[valid_idx])

    assert train_groups.isdisjoint(valid_groups), (
        f"Duplicate leakage detected in fold {fold_number}"
    )

    assert y_train.iloc[train_idx].nunique() == 2
    assert y_train.iloc[valid_idx].nunique() == 2

print("5-fold group-aware CV verified.")
print("No duplicate groups overlap between folds.")

# --------------------------------------------------
# 5. CATBOOST MODEL
# --------------------------------------------------

base_model = CatBoostClassifier(
    loss_function="Logloss",
    eval_metric="PRAUC",
    random_seed=RANDOM_STATE,
    verbose=False,
    allow_writing_files=False,
    thread_count=1
)

# --------------------------------------------------
# 6. HYPERPARAMETER SEARCH
# --------------------------------------------------

param_distributions = {
    "iterations": [200, 300, 400],
    "depth": [4, 5, 6, 7],
    "learning_rate": [0.03, 0.05, 0.1],
    "l2_leaf_reg": [3, 5, 7, 9],
    "random_strength": [0.5, 1, 2],
    "class_weights": [
        [1, 1],
        [1, 100],
        [1, 300],
        [1, 579]
    ]
}

print("\nStarting CatBoost hyperparameter optimisation...")
print(f"Configurations: {N_ITER}")
print(f"Cross-validation folds: {N_SPLITS}")

search = RandomizedSearchCV(
    estimator=base_model,
    param_distributions=param_distributions,
    n_iter=N_ITER,
    scoring="average_precision",
    cv=cv_splits,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=2,
    refit=True,
    return_train_score=False
)

search_start = time.time()

search.fit(X_train, y_train)

search_time = time.time() - search_start

best_model = search.best_estimator_

print("\nOptimisation completed.")
print("Best CV PR-AUC:", round(search.best_score_, 6))
print("Best parameters:")
print(search.best_params_)
print("Search time:", round(search_time, 2), "seconds")

# --------------------------------------------------
# 7. TRAINING-ONLY OUT-OF-FOLD PREDICTIONS
# --------------------------------------------------

print("\nGenerating training-only OOF predictions...")

oof_start = time.time()


print("\nGenerating training-only OOF predictions...")

oof_start = time.time()

oof_probabilities = np.zeros(len(X_train))

for fold_number, (train_idx, valid_idx) in enumerate(
    cv_splits, start=1
):
    print(f"Generating OOF predictions: fold {fold_number}/5")

    fold_model = CatBoostClassifier(
        iterations=400,
        depth=7,
        learning_rate=0.05,
        l2_leaf_reg=9,
        random_strength=2,
        class_weights=[1, 100],
        loss_function="Logloss",
        eval_metric="PRAUC",
        random_seed=RANDOM_STATE,
        verbose=False,
        allow_writing_files=False,
        thread_count=1
    )

    fold_model.fit(
        X_train.iloc[train_idx],
        y_train.iloc[train_idx]
    )

    oof_probabilities[valid_idx] = (
        fold_model.predict_proba(
            X_train.iloc[valid_idx]
        )[:, 1]
    )

oof_time = time.time() - oof_start

print("OOF predictions completed.")


oof_time = time.time() - oof_start

# Select threshold that maximises training OOF F1.
precision_values, recall_values, thresholds = (
    precision_recall_curve(y_train, oof_probabilities)
)

f1_values = (
    2 * precision_values[:-1] * recall_values[:-1]
    / (
        precision_values[:-1] + recall_values[:-1] + 1e-12
    )
)

best_threshold_index = int(np.argmax(f1_values))
best_threshold = float(thresholds[best_threshold_index])

oof_f1 = float(f1_values[best_threshold_index])

print("OOF threshold:", round(best_threshold, 6))
print("OOF F1:", round(oof_f1, 6))
print("OOF prediction time:", round(oof_time, 2), "seconds")

# --------------------------------------------------
# 8. VALIDATION PREDICTIONS
# --------------------------------------------------

print("\nEvaluating on validation data...")

val_probabilities = best_model.predict_proba(X_val)[:, 1]

# --------------------------------------------------
# 9. METRIC CALCULATION
# --------------------------------------------------

def calculate_metrics(y_true, probabilities, threshold):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    ).ravel()

    specificity = (
        tn / (tn + fp) if (tn + fp) > 0 else 0
    )

    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_true, predictions, zero_division=0
        ),
        "pr_auc": average_precision_score(
            y_true, probabilities
        ),
        "roc_auc": roc_auc_score(
            y_true, probabilities
        ),
        "mcc": matthews_corrcoef(
            y_true, predictions
        ),
        "specificity": specificity,
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn)
    }


default_metrics = calculate_metrics(
    y_val,
    val_probabilities,
    0.5
)

oof_threshold_metrics = calculate_metrics(
    y_val,
    val_probabilities,
    best_threshold
)

# --------------------------------------------------
# 10. DISPLAY RESULTS
# --------------------------------------------------

print("\n========== VALIDATION RESULTS ==========")

for name, metrics in [
    ("Default threshold (0.5)", default_metrics),
    ("OOF-selected threshold", oof_threshold_metrics)
]:

    print(f"\n{name}")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")

# --------------------------------------------------
# 11. SAVE RESULTS
# --------------------------------------------------

common_results = {
    "best_cv_pr_auc": search.best_score_,
    "oof_threshold": best_threshold,
    "oof_f1": oof_f1,
    "search_time_seconds": search_time,
    "oof_time_seconds": oof_time,
    "best_parameters": json.dumps(
        search.best_params_,
        default=str
    )
}

rows = []

for name, metrics in [
    ("default_0.5", default_metrics),
    ("oof_selected", oof_threshold_metrics)
]:

    row = {
        "model": "CatBoost Optimised",
        "configuration": name,
        **common_results,
        **metrics
    }

    rows.append(row)

results_df = pd.DataFrame(rows)

results_df.to_csv(
    RESULT_PATH,
    index=False
)

print("\nResults saved successfully:")
print(RESULT_PATH)

print("\n========== EXPERIMENT COMPLETED ==========")
