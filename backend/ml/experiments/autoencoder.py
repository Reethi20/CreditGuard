
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

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

OUTPUT_PATH = OUTPUT_DIR / "autoencoder_results.csv"

RANDOM_STATE = 42

THRESHOLD_PERCENTILE = 99


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
# 3. Prepare legitimate training data
# --------------------------------------------------

def prepare_training_data(X_train, y_train):

    print("\nPreparing legitimate training transactions...")

    # Fraud labels are used only to select legitimate
    # transactions. They are not supplied to the model.

    X_legitimate = X_train.loc[y_train == 0].copy()

    print(
        f"Legitimate training transactions: {len(X_legitimate):,}"
    )

    # Group identical rows together so duplicate transactions
    # cannot cross the fitting/calibration boundary.

    groups = pd.util.hash_pandas_object(
        X_legitimate,
        index=False
    ).to_numpy()

    group_splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=RANDOM_STATE
    )

    fit_indices, calibration_indices = next(
        group_splitter.split(
            X_legitimate,
            groups=groups
        )
    )

    X_fit = X_legitimate.iloc[fit_indices]
    X_calibration = X_legitimate.iloc[calibration_indices]

    print(f"Autoencoder fitting rows: {len(X_fit):,}")
    print(f"Threshold calibration rows: {len(X_calibration):,}")

    return X_fit, X_calibration


# --------------------------------------------------
# 4. Train autoencoder
# --------------------------------------------------

def train_autoencoder(X_fit, X_calibration):

    print("\nScaling training data...")

    scaler = StandardScaler()

    # Fit scaler only on the autoencoder fitting partition.
    X_fit_scaled = scaler.fit_transform(X_fit)

    # Transform calibration data using the fitted scaler.
    X_calibration_scaled = scaler.transform(X_calibration)

    print("\nBuilding autoencoder...")

    model = MLPRegressor(
        hidden_layer_sizes=(16, 8, 16),
        activation="relu",
        solver="adam",
        alpha=0.0001,
        batch_size=256,
        learning_rate_init=0.001,
        max_iter=30,
        early_stopping=True,
        validation_fraction=0.10,
        n_iter_no_change=5,
        random_state=RANDOM_STATE,
        verbose=False,
    )

    print("Training autoencoder on legitimate transactions...")

    start_time = time.time()

    model.fit(X_fit_scaled, X_fit_scaled)

    training_time = time.time() - start_time

    print(f"Training completed in {training_time:.2f} seconds")
    print(f"Training iterations: {model.n_iter_}")

    # Reconstruction error on held-out legitimate training data.
    calibration_reconstruction = model.predict(
        X_calibration_scaled
    )

    calibration_errors = np.mean(
        np.square(
            X_calibration_scaled - calibration_reconstruction
        ),
        axis=1
    )

    # Set threshold using training-only calibration errors.
    threshold = np.percentile(
        calibration_errors,
        THRESHOLD_PERCENTILE
    )

    print(
        f"\nReconstruction error threshold "
        f"({THRESHOLD_PERCENTILE}th percentile): {threshold:.6f}"
    )

    return scaler, model, threshold, training_time


# --------------------------------------------------
# 5. Evaluate autoencoder
# --------------------------------------------------

def evaluate_model(
    scaler,
    model,
    threshold,
    training_time,
    X_val,
    y_val
):

    print("\nEvaluating on validation data...")

    X_val_scaled = scaler.transform(X_val)

    reconstructed = model.predict(X_val_scaled)

    reconstruction_errors = np.mean(
        np.square(X_val_scaled - reconstructed),
        axis=1
    )

    # Higher reconstruction error = more anomalous.
    predictions = (
        reconstruction_errors > threshold
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
        y_val, reconstruction_errors
    )

    roc_auc = roc_auc_score(
        y_val, reconstruction_errors
    )

    mcc = matthews_corrcoef(y_val, predictions)

    tn, fp, fn, tp = confusion_matrix(
        y_val,
        predictions,
        labels=[0, 1]
    ).ravel()

    specificity = (
        tn / (tn + fp) if (tn + fp) else 0
    )

    print("\nAUTOENCODER RESULTS")

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
        "Model": "Autoencoder",
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
        "Architecture": "30-16-8-16-30",
        "Training_Data": "Legitimate transactions only",
    }

    return results


# --------------------------------------------------
# 6. Main execution
# --------------------------------------------------

def main():

    X_train, X_val, y_train, y_val = load_data()

    X_fit, X_calibration = prepare_training_data(
        X_train,
        y_train
    )

    scaler, model, threshold, training_time = train_autoencoder(
        X_fit,
        X_calibration
    )

    results = evaluate_model(
        scaler,
        model,
        threshold,
        training_time,
        X_val,
        y_val
    )

    results_df = pd.DataFrame([results])

    results_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\n----------------------------------------")
    print("AUTOENCODER EXPERIMENT COMPLETED")
    print("----------------------------------------")

    print(results_df.to_string(index=False))

    print(f"\nResults saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
