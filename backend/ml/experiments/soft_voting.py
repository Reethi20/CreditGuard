
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
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

OUTPUT_PATH = OUTPUT_DIR / "soft_voting_results.csv"

RANDOM_STATE = 42


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
# 3. Evaluate soft voting model
# --------------------------------------------------

def evaluate_model(model, X_train, X_val, y_train, y_val):

    print("\nTraining Soft Voting Classifier...")

    start_time = time.time()

    model.fit(X_train, y_train)

    training_time = time.time() - start_time

    # Average probability predictions from all three models.
    probabilities = model.predict_proba(X_val)[:, 1]

    # Standard probability threshold.
    predictions = (probabilities >= 0.5).astype(int)

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

    pr_auc = average_precision_score(y_val, probabilities)

    roc_auc = roc_auc_score(y_val, probabilities)

    mcc = matthews_corrcoef(y_val, predictions)

    tn, fp, fn, tp = confusion_matrix(
        y_val,
        predictions,
        labels=[0, 1]
    ).ravel()

    specificity = tn / (tn + fp) if (tn + fp) else 0

    print("\nSOFT VOTING RESULTS")

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
        "Model": "Soft Voting",
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
        "Threshold": 0.5,
        "Voting": "Equal probability averaging",
    }

    return results


# --------------------------------------------------
# 4. Build base models
# --------------------------------------------------

def build_soft_voting_model():

    logistic_regression = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE,
        ))
    ])

    random_forest = RandomForestClassifier(
        n_estimators=200,
        class_weight=None,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    xgboost = XGBClassifier(
        n_estimators=200,
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

    voting_model = VotingClassifier(
        estimators=[
            ("logistic_regression", logistic_regression),
            ("random_forest", random_forest),
            ("xgboost", xgboost),
        ],
        voting="soft",
        weights=[1, 1, 1],
        n_jobs=1,
    )

    return voting_model


# --------------------------------------------------
# 5. Main execution
# --------------------------------------------------

def main():

    X_train, X_val, y_train, y_val = load_data()

    model = build_soft_voting_model()

    results = evaluate_model(
        model,
        X_train,
        X_val,
        y_train,
        y_val,
    )

    results_df = pd.DataFrame([results])

    results_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\n----------------------------------------")
    print("SOFT VOTING EXPERIMENT COMPLETED")
    print("----------------------------------------")

    print(results_df.to_string(index=False))

    print(f"\nResults saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
