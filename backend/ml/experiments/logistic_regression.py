
from pathlib import Path
import time

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix
)


# ==================================================
# 1. PROJECT PATHS
# ==================================================

ROOT = Path(__file__).resolve().parents[3]

RAW_DATA_PATH = ROOT / "data" / "raw" / "creditcard.csv"

SPLIT_PATH = (
    ROOT / "data" / "processed" / "experiment_splits.csv"
)

RESULTS_DIR = ROOT / "models" / "experiments"

RESULTS_PATH = (
    RESULTS_DIR / "logistic_regression_results.csv"
)


# ==================================================
# 2. LOAD DATA
# ==================================================

def load_data():

    print("\nLoading original dataset...")

    df = pd.read_csv(RAW_DATA_PATH)

    print(f"Dataset shape: {df.shape}")
    print(f"Fraud cases: {df['Class'].sum()}")
    print(f"Legitimate cases: {(df['Class'] == 0).sum()}")

    return df


# ==================================================
# 3. LOAD SAVED SPLITS
# ==================================================

def load_splits():

    print("\nLoading saved experiment splits...")

    splits = pd.read_csv(SPLIT_PATH)

    return splits


# ==================================================
# 4. VERIFY DATA INTEGRITY
# ==================================================

def verify_data(df, splits):

    print("\nVerifying data integrity...")

    if len(df) != len(splits):
        raise ValueError("Dataset and split counts do not match!")

    if splits["row_index"].duplicated().any():
        raise ValueError("Duplicate row indices detected!")

    expected_splits = {"train", "validation", "test"}

    if set(splits["split"].unique()) != expected_splits:
        raise ValueError("Unexpected split labels detected!")

    if sorted(splits["row_index"].tolist()) != list(range(len(df))):
        raise ValueError("Missing or invalid row indices!")

    print("Data integrity checks passed.")


# ==================================================
# 5. PREPARE TRAINING AND VALIDATION DATA
# ==================================================

def prepare_data(df, splits):

    train_idx = splits.loc[
        splits["split"] == "train", "row_index"
    ].to_numpy()

    validation_idx = splits.loc[
        splits["split"] == "validation", "row_index"
    ].to_numpy()

    X_train = df.iloc[train_idx].drop(columns=["Class"])
    y_train = df.iloc[train_idx]["Class"]

    X_validation = df.iloc[validation_idx].drop(columns=["Class"])
    y_validation = df.iloc[validation_idx]["Class"]

    print("\nPrepared datasets:")
    print(f"Training records: {len(X_train)}")
    print(f"Validation records: {len(X_validation)}")

    print(f"Training fraud cases: {y_train.sum()}")
    print(f"Validation fraud cases: {y_validation.sum()}")

    return X_train, y_train, X_validation, y_validation


# ==================================================
# 6. EVALUATE MODEL
# ==================================================

def evaluate_model(
    model,
    X_validation,
    y_validation,
    experiment_name,
    training_time
):

    predictions = model.predict(X_validation)

    probabilities = model.predict_proba(X_validation)[:, 1]

    precision = precision_score(
        y_validation, predictions, zero_division=0
    )

    recall = recall_score(
        y_validation, predictions, zero_division=0
    )

    f1 = f1_score(
        y_validation, predictions, zero_division=0
    )

    pr_auc = average_precision_score(
        y_validation, probabilities
    )

    roc_auc = roc_auc_score(
        y_validation, probabilities
    )

    mcc = matthews_corrcoef(
        y_validation, predictions
    )

    tn, fp, fn, tp = confusion_matrix(
        y_validation,
        predictions,
        labels=[0, 1]
    ).ravel()

    specificity = tn / (tn + fp) if (tn + fp) else 0

    result = {
        "experiment": experiment_name,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "mcc": mcc,
        "specificity": specificity,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "training_time_seconds": training_time
    }

    print(f"\nResults: {experiment_name}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"MCC: {mcc:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"True Positives: {tp}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Negatives: {tn}")

    return result


# ==================================================
# 7. RUN LOGISTIC REGRESSION EXPERIMENTS
# ==================================================

def run_experiments(
    X_train,
    y_train,
    X_validation,
    y_validation
):

    experiments = {
        "Logistic Regression - Baseline": None,
        "Logistic Regression - Class Weighted": "balanced"
    }

    results = []

    for experiment_name, class_weight in experiments.items():

        print(f"\nRunning {experiment_name}...")

        pipeline = Pipeline([
            ("scaler", StandardScaler()),

            ("model", LogisticRegression(
                class_weight=class_weight,
                max_iter=2000,
                random_state=42
            ))
        ])

        start_time = time.perf_counter()

        pipeline.fit(X_train, y_train)

        training_time = time.perf_counter() - start_time

        result = evaluate_model(
            pipeline,
            X_validation,
            y_validation,
            experiment_name,
            training_time
        )

        results.append(result)

    return pd.DataFrame(results)


# ==================================================
# 8. MAIN EXECUTION
# ==================================================

def main():

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()

    splits = load_splits()

    verify_data(df, splits)

    (
        X_train,
        y_train,
        X_validation,
        y_validation
    ) = prepare_data(df, splits)

    results = run_experiments(
        X_train,
        y_train,
        X_validation,
        y_validation
    )

    results.to_csv(
        RESULTS_PATH,
        index=False
    )

    print("\n========== FINAL RESULTS ==========")

    print(results.to_string(index=False))

    print("\nResults saved to:")
    print(RESULTS_PATH)

    print("\nTest dataset was not used.")
    print("Original raw dataset remains unchanged.")
    print("\nLOGISTIC REGRESSION EXPERIMENT COMPLETED!")


if __name__ == "__main__":
    main()
