
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
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

OUTPUT_PATH = OUTPUT_DIR / "hybrid_anomaly_results.csv"

RANDOM_STATE = 42

N_ESTIMATORS = 200

CALIBRATION_SIZE = 0.20

THRESHOLD_PERCENTILE = 99

XGBOOST_WEIGHT = 0.5
ISOLATION_FOREST_WEIGHT = 0.5


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
        raise ValueError(
            "Split file must contain row_index and split columns."
        )

    if splits["row_index"].duplicated().any():
        raise ValueError("Duplicate row indices found in split file.")

    if not np.array_equal(
        np.sort(splits["row_index"].to_numpy()),
        np.arange(len(df))
    ):
        raise ValueError(
            "Split file does not cover every original row exactly once."
        )

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

    X_train = X.iloc[train_indices].reset_index(drop=True)
    y_train = y.iloc[train_indices].reset_index(drop=True)

    X_val = X.iloc[val_indices].reset_index(drop=True)
    y_val = y.iloc[val_indices].reset_index(drop=True)

    print(f"Training rows: {len(X_train):,}")
    print(f"Validation rows: {len(X_val):,}")
    print(f"Training fraud cases: {int(y_train.sum())}")
    print(f"Validation fraud cases: {int(y_val.sum())}")

    return X_train, X_val, y_train, y_val


# --------------------------------------------------
# 3. Create leakage-safe fitting/calibration split
# --------------------------------------------------

def prepare_training_partitions(X_train, y_train):

    print("\nCreating fitting and calibration partitions...")

    # Identical feature rows stay in the same partition.
    groups = pd.util.hash_pandas_object(
        X_train,
        index=False
    ).to_numpy()

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=CALIBRATION_SIZE,
        random_state=RANDOM_STATE
    )

    fit_indices, calibration_indices = next(
        splitter.split(
            X_train,
            y_train,
            groups=groups
        )
    )

    X_fit = X_train.iloc[fit_indices].reset_index(drop=True)
    y_fit = y_train.iloc[fit_indices].reset_index(drop=True)

    X_calibration = X_train.iloc[
        calibration_indices
    ].reset_index(drop=True)

    y_calibration = y_train.iloc[
        calibration_indices
    ].reset_index(drop=True)

    fit_groups = set(groups[fit_indices])
    calibration_groups = set(groups[calibration_indices])

    if fit_groups.intersection(calibration_groups):
        raise ValueError("Duplicate groups overlap between partitions.")

    # The calibration partition must contain legitimate transactions.
    X_calibration_legitimate = X_calibration.loc[
        y_calibration == 0
    ].copy()

    if len(X_calibration_legitimate) == 0:
        raise ValueError("No legitimate calibration transactions found.")

    print(f"Model fitting rows: {len(X_fit):,}")
    print(f"Model fitting fraud cases: {int(y_fit.sum())}")

    print(
        "Legitimate calibration rows: "
        f"{len(X_calibration_legitimate):,}"
    )

    print(
        "Calibration fraud labels are not used for "
        "threshold selection."
    )

    return (
        X_fit,
        y_fit,
        X_calibration_legitimate
    )


# --------------------------------------------------
# 4. Build and train both models
# --------------------------------------------------

def train_models(X_fit, y_fit):

    print("\nBuilding XGBoost...")

    xgboost = XGBClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=6,
        learning_rate=0.1,
        subsample=1.0,
        colsample_bytree=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
    )

    print("Building Isolation Forest...")

    isolation_forest = IsolationForest(
        n_estimators=N_ESTIMATORS,
        max_samples="auto",
        contamination="auto",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    X_fit_legitimate = X_fit.loc[y_fit == 0]

    if len(X_fit_legitimate) == 0:
        raise ValueError("No legitimate fitting transactions found.")

    print(f"Legitimate rows for Isolation Forest: {len(X_fit_legitimate):,}")

    print("\nTraining both models...")

    start_time = time.time()

    # XGBoost learns from labelled fitting transactions.
    xgboost.fit(X_fit, y_fit)

    # Isolation Forest learns only from legitimate fitting transactions.
    isolation_forest.fit(X_fit_legitimate)

    training_time = time.time() - start_time

    print(f"Model training time: {training_time:.2f} seconds")

    return xgboost, isolation_forest, training_time


# --------------------------------------------------
# 5. Generate model scores
# --------------------------------------------------

def generate_scores(xgboost, isolation_forest, X):

    # XGBoost fraud probability.
    xgb_scores = xgboost.predict_proba(X)[:, 1]

    # Higher Isolation Forest scores indicate more anomalous records.
    isolation_scores = -isolation_forest.decision_function(X)

    return xgb_scores, isolation_scores


# --------------------------------------------------
# 6. Normalize scores using calibration data only
# --------------------------------------------------

def percentile_scores(scores, reference_scores):

    sorted_reference = np.sort(
        np.asarray(reference_scores)
    )

    ranks = np.searchsorted(
        sorted_reference,
        scores,
        side="right"
    )

    return ranks / len(sorted_reference)


def create_hybrid_scores(
    xgb_scores,
    isolation_scores,
    xgb_reference,
    isolation_reference
):

    normalized_xgb = percentile_scores(
        xgb_scores,
        xgb_reference
    )

    normalized_isolation = percentile_scores(
        isolation_scores,
        isolation_reference
    )

    hybrid_scores = (
        XGBOOST_WEIGHT * normalized_xgb
        +
        ISOLATION_FOREST_WEIGHT * normalized_isolation
    )

    return hybrid_scores


# --------------------------------------------------
# 7. Calibrate decision threshold
# --------------------------------------------------

def calibrate_threshold(
    xgboost,
    isolation_forest,
    X_calibration
):

    print("\nCalibrating hybrid score...")

    # Calibration data contains legitimate transactions only.
    xgb_calibration, isolation_calibration = generate_scores(
        xgboost,
        isolation_forest,
        X_calibration
    )

    # These reference distributions are used to normalize
    # both calibration and validation scores.
    hybrid_calibration = create_hybrid_scores(
        xgb_calibration,
        isolation_calibration,
        xgb_calibration,
        isolation_calibration
    )

    threshold = np.percentile(
        hybrid_calibration,
        THRESHOLD_PERCENTILE
    )

    print(
        f"Hybrid threshold ({THRESHOLD_PERCENTILE}th percentile): "
        f"{threshold:.6f}"
    )

    return (
        threshold,
        xgb_calibration,
        isolation_calibration
    )


# --------------------------------------------------
# 8. Evaluate hybrid model
# --------------------------------------------------

def evaluate_model(
    xgboost,
    isolation_forest,
    X_val,
    y_val,
    threshold,
    xgb_reference,
    isolation_reference,
    training_time
):

    print("\nEvaluating hybrid model on validation data...")

    xgb_val_scores, isolation_val_scores = generate_scores(
        xgboost,
        isolation_forest,
        X_val
    )

    hybrid_scores = create_hybrid_scores(
        xgb_val_scores,
        isolation_val_scores,
        xgb_reference,
        isolation_reference
    )

    predictions = (
        hybrid_scores > threshold
    ).astype(int)

    accuracy = accuracy_score(y_val, predictions)

    precision = precision_score(
        y_val, predictions, zero_division=0
    )

    recall = recall_score(
        y_val, predictions, zero_division=0
    )

    f1 = f1_score(
        y_val, predictions, zero_division=0
    )

    pr_auc = average_precision_score(
        y_val, hybrid_scores
    )

    roc_auc = roc_auc_score(
        y_val, hybrid_scores
    )

    mcc = matthews_corrcoef(
        y_val, predictions
    )

    tn, fp, fn, tp = confusion_matrix(
        y_val,
        predictions,
        labels=[0, 1]
    ).ravel()

    specificity = (
        tn / (tn + fp) if (tn + fp) else 0
    )

    print("\nHYBRID ANOMALY RESULTS")

    print(f"Accuracy:    {accuracy:.4%}")
    print(f"Precision:   {precision:.4f}")
    print(f"Recall:      {recall:.4f}")
    print(f"F1 Score:    {f1:.4f}")
    print(f"PR-AUC:      {pr_auc:.4f}")
    print(f"ROC-AUC:     {roc_auc:.4f}")
    print(f"MCC:         {mcc:.4f}")
    print(f"Specificity: {specificity:.4f}")

    print(f"TP: {tp} | FP: {fp} | TN: {tn} | FN: {fn}")
    print(f"Training time: {training_time:.2f} seconds")

    results = {
        "Model": "XGBoost + Isolation Forest Hybrid",
        "Accuracy": accuracy,
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
        "Threshold": threshold,
        "Threshold_Percentile": THRESHOLD_PERCENTILE,
        "XGBoost_Weight": XGBOOST_WEIGHT,
        "Isolation_Forest_Weight": ISOLATION_FOREST_WEIGHT,
        "Calibration_Data": "Legitimate training transactions only",
    }

    return results


# --------------------------------------------------
# 9. Main execution
# --------------------------------------------------

def main():

    X_train, X_val, y_train, y_val = load_data()

    (
        X_fit,
        y_fit,
        X_calibration
    ) = prepare_training_partitions(
        X_train,
        y_train
    )

    xgboost, isolation_forest, training_time = train_models(
        X_fit,
        y_fit
    )

    (
        threshold,
        xgb_reference,
        isolation_reference
    ) = calibrate_threshold(
        xgboost,
        isolation_forest,
        X_calibration
    )

    results = evaluate_model(
        xgboost,
        isolation_forest,
        X_val,
        y_val,
        threshold,
        xgb_reference,
        isolation_reference,
        training_time
    )

    results_df = pd.DataFrame([results])

    results_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\n----------------------------------------")
    print("HYBRID ANOMALY EXPERIMENT COMPLETED")
    print("----------------------------------------")

    print(results_df.to_string(index=False))

    print(f"\nResults saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
