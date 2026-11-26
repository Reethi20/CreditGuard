
import time
from pathlib import Path

import numpy as np
import pandas as pd

from xgboost import XGBClassifier

from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedGroupKFold,
    cross_val_predict,
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
    precision_recall_curve,
)


# --------------------------------------------------
# 1. Project paths and configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"
SPLIT_PATH = PROJECT_ROOT / "data" / "processed" / "experiment_splits.csv"

OUTPUT_DIR = PROJECT_ROOT / "models" / "experiments"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "xgboost_optimization_results.csv"

RANDOM_STATE = 42
CV_FOLDS = 5
N_ITER = 8

DEFAULT_THRESHOLD = 0.5


# --------------------------------------------------
# 2. Load and validate data
# --------------------------------------------------

def load_data():

    print("Loading original dataset...")

    df = pd.read_csv(DATA_PATH)
    splits = pd.read_csv(SPLIT_PATH)

    if len(df) != len(splits):
        raise ValueError("Dataset and split row counts do not match.")

    if not {"row_index", "split"}.issubset(splits.columns):
        raise ValueError("Invalid split file columns.")

    if splits["row_index"].duplicated().any():
        raise ValueError("Duplicate row indices found.")

    if not np.array_equal(
        np.sort(splits["row_index"].to_numpy()),
        np.arange(len(df))
    ):
        raise ValueError("Split file does not cover every original row.")

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

    if y_train.nunique() != 2 or y_val.nunique() != 2:
        raise ValueError("Both partitions must contain both classes.")

    return X_train, X_val, y_train, y_val


# --------------------------------------------------
# 3. Create group-aware CV folds
# --------------------------------------------------

def create_cv_folds(X_train, y_train):

    print("\nCreating group-aware CV folds...")

    groups = pd.util.hash_pandas_object(
        X_train,
        index=False
    ).to_numpy()

    cv = StratifiedGroupKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    cv_splits = list(
        cv.split(X_train, y_train, groups)
    )

    for fold_number, (train_idx, valid_idx) in enumerate(
        cv_splits, start=1
    ):

        train_groups = set(groups[train_idx])
        valid_groups = set(groups[valid_idx])

        if train_groups.intersection(valid_groups):
            raise ValueError(
                f"Duplicate group leakage in fold {fold_number}."
            )

        if y_train.iloc[train_idx].nunique() != 2:
            raise ValueError(
                f"Training fold {fold_number} lacks a class."
            )

        if y_train.iloc[valid_idx].nunique() != 2:
            raise ValueError(
                f"Validation fold {fold_number} lacks a class."
            )

        print(
            f"Fold {fold_number}: "
            f"train={len(train_idx):,}, "
            f"validation={len(valid_idx):,}, "
            f"fraud={int(y_train.iloc[valid_idx].sum())}"
        )

    print("Duplicate group separation verified.")

    return cv_splits


# --------------------------------------------------
# 4. Define hyperparameter search space
# --------------------------------------------------

def get_parameter_space():

    return {
        "n_estimators": [200, 300, 400],
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.03, 0.05, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "min_child_weight": [1, 3, 5],
        "scale_pos_weight": [1, 100, 300, 579],
    }


# --------------------------------------------------
# 5. Optimise XGBoost
# --------------------------------------------------

def optimise_model(X_train, y_train, cv_splits):

    print("\nStarting XGBoost optimisation...")

    print(f"Configurations sampled: {N_ITER}")
    print(f"Cross-validation folds: {CV_FOLDS}")
    print(f"Total model fits: {N_ITER * CV_FOLDS} plus final refit")

    base_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=1,
        tree_method="hist",
        verbosity=0
    )

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=get_parameter_space(),
        n_iter=N_ITER,
        scoring="average_precision",
        cv=cv_splits,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=2,
        refit=True,
        return_train_score=False
    )

    start_time = time.time()

    search.fit(X_train, y_train)

    optimisation_time = time.time() - start_time

    print("\nOptimisation completed.")

    print(f"Best CV PR-AUC: {search.best_score_:.6f}")

    print("\nBest hyperparameters:")

    for parameter, value in search.best_params_.items():
        print(f"{parameter}: {value}")

    print(f"\nOptimisation time: {optimisation_time:.2f} seconds")

    return (
        search.best_estimator_,
        search.best_params_,
        search.best_score_,
        optimisation_time
    )


# --------------------------------------------------
# 6. Select threshold using OOF predictions
# --------------------------------------------------

def select_threshold(best_model, X_train, y_train, cv_splits):

    print("\nGenerating out-of-fold training predictions...")

    start_time = time.time()

    oof_probabilities = cross_val_predict(
        best_model,
        X_train,
        y_train,
        cv=cv_splits,
        method="predict_proba",
        n_jobs=1
    )[:, 1]

    oof_time = time.time() - start_time

    precision, recall, thresholds = precision_recall_curve(
        y_train,
        oof_probabilities
    )

    f1_scores = (
        2 * precision[:-1] * recall[:-1]
        /
        (precision[:-1] + recall[:-1] + 1e-12)
    )

    best_index = int(np.argmax(f1_scores))

    selected_threshold = float(thresholds[best_index])

    print(f"Selected threshold: {selected_threshold:.6f}")
    print(f"OOF F1: {f1_scores[best_index]:.4f}")
    print(f"OOF prediction time: {oof_time:.2f} seconds")

    return selected_threshold, oof_time


# --------------------------------------------------
# 7. Evaluate model
# --------------------------------------------------

def evaluate_model(
    name,
    X_val,
    y_val,
    probabilities,
    threshold,
    best_params,
    cv_pr_auc,
    optimisation_time,
    oof_time
):

    predictions = (
        probabilities >= threshold
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
        y_val, probabilities
    )

    roc_auc = roc_auc_score(
        y_val, probabilities
    )

    mcc = matthews_corrcoef(
        y_val, predictions
    )

    tn, fp, fn, tp = confusion_matrix(
        y_val,
        predictions,
        labels=[0, 1]
    ).ravel()

    specificity = tn / (tn + fp) if (tn + fp) else 0

    result = {
        "Model": name,
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
        "Threshold": threshold,
        "CV_PR_AUC": cv_pr_auc,
        "Optimisation_Time_Seconds": optimisation_time,
        "OOF_Time_Seconds": oof_time,
        "Best_Parameters": str(best_params)
    }

    print(f"\n{name}")
    print("-" * 45)
    print(f"Threshold:   {threshold:.6f}")
    print(f"Accuracy:    {accuracy:.4%}")
    print(f"Precision:   {precision:.4f}")
    print(f"Recall:      {recall:.4f}")
    print(f"F1 Score:    {f1:.4f}")
    print(f"PR-AUC:      {pr_auc:.4f}")
    print(f"ROC-AUC:     {roc_auc:.4f}")
    print(f"MCC:         {mcc:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"TP: {tp} | FP: {fp} | TN: {tn} | FN: {fn}")

    return result


# --------------------------------------------------
# 8. Main execution
# --------------------------------------------------

def main():

    X_train, X_val, y_train, y_val = load_data()

    cv_splits = create_cv_folds(
        X_train,
        y_train
    )

    (
        best_model,
        best_params,
        cv_pr_auc,
        optimisation_time
    ) = optimise_model(
        X_train,
        y_train,
        cv_splits
    )

    selected_threshold, oof_time = select_threshold(
        best_model,
        X_train,
        y_train,
        cv_splits
    )

    print("\nGenerating validation probabilities...")

    probabilities = best_model.predict_proba(X_val)[:, 1]

    results = []

    results.append(
        evaluate_model(
            "XGBoost Optimised - Default Threshold",
            X_val,
            y_val,
            probabilities,
            DEFAULT_THRESHOLD,
            best_params,
            cv_pr_auc,
            optimisation_time,
            oof_time
        )
    )

    results.append(
        evaluate_model(
            "XGBoost Optimised - OOF Threshold",
            X_val,
            y_val,
            probabilities,
            selected_threshold,
            best_params,
            cv_pr_auc,
            optimisation_time,
            oof_time
        )
    )

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\n----------------------------------------")
    print("XGBOOST OPTIMISATION COMPLETED")
    print("----------------------------------------")

    print(
        results_df[
            [
                "Model",
                "Accuracy",
                "Precision",
                "Recall",
                "F1",
                "PR_AUC",
                "ROC_AUC",
                "Threshold",
                "TP",
                "FP",
                "TN",
                "FN"
            ]
        ].to_string(index=False)
    )

    print(f"\nResults saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
