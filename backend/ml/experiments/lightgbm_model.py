
import time
from pathlib import Path

import numpy as np
import pandas as pd

from lightgbm import LGBMClassifier

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
)


# --------------------------------------------------
# 1. Project paths and configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"
SPLIT_PATH = PROJECT_ROOT / "data" / "processed" / "experiment_splits.csv"

OUTPUT_DIR = PROJECT_ROOT / "models" / "experiments"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "lightgbm_results.csv"

RANDOM_STATE = 42
N_ESTIMATORS = 200


# --------------------------------------------------
# 2. Load and validate data
# --------------------------------------------------

def load_data():

    print("Loading original dataset...")

    df = pd.read_csv(DATA_PATH)
    splits = pd.read_csv(SPLIT_PATH)

    if len(df) != len(splits):
        raise ValueError("Dataset and split file row counts do not match.")

    if "Class" not in df.columns:
        raise ValueError("Target column 'Class' is missing.")

    if not {"row_index", "split"}.issubset(splits.columns):
        raise ValueError("Split file must contain row_index and split columns.")

    if splits["row_index"].duplicated().any():
        raise ValueError("Duplicate row indices found in split file.")

    if not np.array_equal(
        np.sort(splits["row_index"].to_numpy()),
        np.arange(len(df))
    ):
        raise ValueError("Split file does not cover every original row exactly once.")

    if not set(splits["split"].unique()).issubset(
        {"train", "validation", "test"}
    ):
        raise ValueError("Unexpected split label found.")

    train_indices = splits.loc[
        splits["split"] == "train", "row_index"
    ].to_numpy(dtype=int)

    val_indices = splits.loc[
        splits["split"] == "validation", "row_index"
    ].to_numpy(dtype=int)

    X = df.drop(columns=["Class"])
    y = df["Class"]

    X_train = X.iloc[train_indices]
    y_train = y.iloc[train_indices]

    X_val = X.iloc[val_indices]
    y_val = y.iloc[val_indices]

    print(f"Training rows: {len(X_train):,}")
    print(f"Validation rows: {len(X_val):,}")
    print(f"Training fraud cases: {int(y_train.sum())}")
    print(f"Validation fraud cases: {int(y_val.sum())}")

    return X_train, X_val, y_train, y_val


# --------------------------------------------------
# 3. Evaluate model
# --------------------------------------------------

def evaluate_model(name, model, X_train, X_val, y_train, y_val):

    print(f"\nTraining: {name}")

    start_time = time.time()

    model.fit(X_train, y_train)

    training_time = time.time() - start_time

    predictions = model.predict(X_val)
    probabilities = model.predict_proba(X_val)[:, 1]

    precision = precision_score(
        y_val, predictions, zero_division=0
    )

    recall = recall_score(
        y_val, predictions, zero_division=0
    )

    f1 = f1_score(
        y_val, predictions, zero_division=0
    )

    pr_auc = average_precision_score(y_val, probabilities)

    roc_auc = roc_auc_score(y_val, probabilities)

    mcc = matthews_corrcoef(y_val, predictions)

    tn, fp, fn, tp = confusion_matrix(
        y_val, predictions, labels=[0, 1]
    ).ravel()

    specificity = tn / (tn + fp) if (tn + fp) else 0

    results = {
        "Model": name,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "PR_AUC": pr_auc,
        "ROC_AUC": roc_auc,
        "MCC": mcc,
        "Specificity": specificity,
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "Training_Time_Seconds": training_time,
    }

    print(f"Precision:   {precision:.4f}")
    print(f"Recall:      {recall:.4f}")
    print(f"F1 Score:    {f1:.4f}")
    print(f"PR-AUC:      {pr_auc:.4f}")
    print(f"ROC-AUC:     {roc_auc:.4f}")
    print(f"MCC:         {mcc:.4f}")
    print(f"Specificity: {specificity:.4f}")

    print(f"TP: {tp} | FP: {fp} | TN: {tn} | FN: {fn}")
    print(f"Training time: {training_time:.2f} seconds")

    return results


# --------------------------------------------------
# 4. Run LightGBM experiments
# --------------------------------------------------

def main():

    X_train, X_val, y_train, y_val = load_data()

    # Calculate class imbalance using training data only.
    negative_count = int((y_train == 0).sum())
    positive_count = int((y_train == 1).sum())

    scale_pos_weight = negative_count / positive_count

    print(f"\nCalculated scale_pos_weight: {scale_pos_weight:.2f}")

    experiments = [
        (
            "LightGBM - Baseline",
            1
        ),
        (
            "LightGBM - Class Weighted",
            scale_pos_weight
        ),
    ]

    all_results = []

    for name, positive_weight in experiments:

        model = LGBMClassifier(
            n_estimators=N_ESTIMATORS,
            learning_rate=0.1,
            num_leaves=31,
            max_depth=-1,
            objective="binary",
            scale_pos_weight=positive_weight,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=-1,
        )

        results = evaluate_model(
            name,
            model,
            X_train,
            X_val,
            y_train,
            y_val,
        )

        all_results.append(results)

    results_df = pd.DataFrame(all_results)

    results_df.to_csv(OUTPUT_PATH, index=False)

    print("\n----------------------------------------")
    print("LIGHTGBM EXPERIMENT COMPLETED")
    print("----------------------------------------")

    print(results_df.to_string(index=False))

    print(f"\nResults saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
