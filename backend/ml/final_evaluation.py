
import json
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
    classification_report,
    roc_curve,
    precision_recall_curve
)


# ==================================================
# 1. PROJECT PATHS AND CONFIGURATION
# ==================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data/raw/creditcard.csv"
SPLIT_PATH = ROOT / "data/processed/experiment_splits.csv"

OUTPUT_DIR = ROOT / "models/final"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = OUTPUT_DIR / "final_xgboost_model.joblib"
METRICS_PATH = OUTPUT_DIR / "final_test_metrics.csv"
PREDICTIONS_PATH = OUTPUT_DIR / "final_test_predictions.csv"
REPORT_PATH = OUTPUT_DIR / "classification_report.txt"

CONFUSION_PATH = OUTPUT_DIR / "confusion_matrix.png"
ROC_PATH = OUTPUT_DIR / "roc_curve.png"
PR_PATH = OUTPUT_DIR / "precision_recall_curve.png"

THRESHOLD = 0.5
RANDOM_STATE = 42


# ==================================================
# 2. LOAD AND VERIFY DATA
# ==================================================

print("\nLoading original dataset...")

df = pd.read_csv(DATA_PATH)
splits = pd.read_csv(SPLIT_PATH)

if len(df) != len(splits):
    raise ValueError("Dataset and split row counts do not match.")

if not {"row_index", "split"}.issubset(splits.columns):
    raise ValueError("Unexpected split file structure.")

if splits["row_index"].duplicated().any():
    raise ValueError("Duplicate row indices found in split file.")

if not np.array_equal(
    np.sort(splits["row_index"].to_numpy()),
    np.arange(len(df))
):
    raise ValueError("Split file does not cover every dataset row.")

if splits["split"].isna().any():
    raise ValueError("Missing split labels found.")

allowed_splits = {"train", "validation", "test"}

if set(splits["split"].unique()) != allowed_splits:
    raise ValueError(
        f"Expected split labels {allowed_splits}, "
        f"found {set(splits['split'].unique())}"
    )

train_indices = splits.loc[
    splits["split"] == "train", "row_index"
].to_numpy(dtype=int)

val_indices = splits.loc[
    splits["split"] == "validation", "row_index"
].to_numpy(dtype=int)

test_indices = splits.loc[
    splits["split"] == "test", "row_index"
].to_numpy(dtype=int)

development_indices = np.concatenate(
    [train_indices, val_indices]
)

X = df.drop(columns=["Class"])
y = df["Class"]

X_development = X.iloc[
    development_indices
].reset_index(drop=True)

y_development = y.iloc[
    development_indices
].reset_index(drop=True)

X_test = X.iloc[
    test_indices
].reset_index(drop=True)

y_test = y.iloc[
    test_indices
].reset_index(drop=True)

print("\nPartition verification:")
print("Original dataset:", len(df))
print("Training rows:", len(train_indices))
print("Validation rows:", len(val_indices))
print("Development rows:", len(X_development))
print("Test rows:", len(X_test))

print("\nFraud cases:")
print("Development:", int(y_development.sum()))
print("Test:", int(y_test.sum()))

if len(X_development) != 227845:
    raise ValueError("Unexpected development partition size.")

if len(X_test) != 56962:
    raise ValueError("Unexpected test partition size.")

if int(y_test.sum()) != 99:
    raise ValueError("Unexpected number of test fraud cases.")

if y_development.nunique() != 2 or y_test.nunique() != 2:
    raise ValueError("Both partitions must contain both classes.")


# ==================================================
# 3. VERIFY DUPLICATE GROUP SEPARATION
# ==================================================

print("\nChecking duplicate-group separation...")

development_groups = pd.util.hash_pandas_object(
    X_development,
    index=False
).to_numpy()

test_groups = pd.util.hash_pandas_object(
    X_test,
    index=False
).to_numpy()

development_group_set = set(development_groups)
test_group_set = set(test_groups)

overlap = development_group_set.intersection(
    test_group_set
)

if overlap:
    raise ValueError(
        "Duplicate feature groups overlap between "
        "development and test partitions."
    )

print("No identical feature groups overlap.")
print("Test partition remains independent.")


# ==================================================
# 4. DEFINE FINAL XGBOOST MODEL
# ==================================================

print("\nConfiguring final XGBoost model...")

model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    n_estimators=400,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=1.0,
    min_child_weight=5,
    scale_pos_weight=300,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    tree_method="hist",
    verbosity=0
)


# ==================================================
# 5. TRAIN FINAL MODEL
# ==================================================

print("\nTraining final model on development data...")

start_time = time.time()

model.fit(
    X_development,
    y_development
)

training_time = time.time() - start_time

print(
    f"Final training completed in "
    f"{training_time:.2f} seconds."
)


# ==================================================
# 6. SAVE FINAL MODEL
# ==================================================

joblib.dump(model, MODEL_PATH)

print("\nFinal model saved:")
print(MODEL_PATH)


# ==================================================
# 7. ONE-TIME TEST EVALUATION
# ==================================================

print("\nEvaluating on untouched test partition...")

test_probabilities = model.predict_proba(X_test)[:, 1]

test_predictions = (
    test_probabilities >= THRESHOLD
).astype(int)

tn, fp, fn, tp = confusion_matrix(
    y_test,
    test_predictions,
    labels=[0, 1]
).ravel()

accuracy = accuracy_score(y_test, test_predictions)

precision = precision_score(
    y_test,
    test_predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    test_predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    test_predictions,
    zero_division=0
)

pr_auc = average_precision_score(
    y_test,
    test_probabilities
)

roc_auc = roc_auc_score(
    y_test,
    test_probabilities
)

mcc = matthews_corrcoef(
    y_test,
    test_predictions
)

specificity = (
    tn / (tn + fp) if (tn + fp) > 0 else 0
)


# ==================================================
# 8. DISPLAY FINAL METRICS
# ==================================================

metrics = {
    "Model": "Optimised XGBoost",
    "Threshold": THRESHOLD,
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1": f1,
    "PR_AUC": pr_auc,
    "ROC_AUC": roc_auc,
    "MCC": mcc,
    "Specificity": specificity,
    "TP": int(tp),
    "FP": int(fp),
    "TN": int(tn),
    "FN": int(fn),
    "Training_Time_Seconds": training_time,
    "Development_Rows": len(X_development),
    "Test_Rows": len(X_test)
}

print("\n======================================")
print("       FINAL TEST PERFORMANCE")
print("======================================")

for name, value in metrics.items():
    if isinstance(value, float):
        print(f"{name}: {value:.6f}")
    else:
        print(f"{name}: {value}")

print("\nConfusion Matrix:")
print(confusion_matrix(
    y_test,
    test_predictions,
    labels=[0, 1]
))

print("\nClassification Report:")
report = classification_report(
    y_test,
    test_predictions,
    target_names=["Legitimate", "Fraud"],
    digits=4,
    zero_division=0
)

print(report)


# ==================================================
# 9. SAVE METRICS AND PREDICTIONS
# ==================================================

pd.DataFrame([metrics]).to_csv(
    METRICS_PATH,
    index=False
)

prediction_results = pd.DataFrame({
    "row_index": test_indices,
    "actual_class": y_test,
    "fraud_probability": test_probabilities,
    "predicted_class": test_predictions
})

prediction_results.to_csv(
    PREDICTIONS_PATH,
    index=False
)

with open(REPORT_PATH, "w") as file:
    file.write(report)

print("\nMetrics saved:", METRICS_PATH)
print("Predictions saved:", PREDICTIONS_PATH)
print("Classification report saved:", REPORT_PATH)


# ==================================================
# 10. CONFUSION MATRIX FIGURE
# ==================================================

cm = np.array([
    [tn, fp],
    [fn, tp]
])

plt.figure(figsize=(7, 6))

plt.imshow(cm, interpolation="nearest")
plt.title("Final XGBoost Confusion Matrix")
plt.colorbar()

plt.xticks([0, 1], ["Legitimate", "Fraud"])
plt.yticks([0, 1], ["Legitimate", "Fraud"])

plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")

for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center"
        )

plt.tight_layout()
plt.savefig(CONFUSION_PATH, dpi=300)
plt.close()


# ==================================================
# 11. ROC CURVE
# ==================================================

fpr, tpr, _ = roc_curve(
    y_test,
    test_probabilities
)

plt.figure(figsize=(7, 6))

plt.plot(
    fpr,
    tpr,
    label=f"XGBoost (AUC = {roc_auc:.4f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Final XGBoost ROC Curve")
plt.legend()
plt.tight_layout()

plt.savefig(ROC_PATH, dpi=300)
plt.close()


# ==================================================
# 12. PRECISION-RECALL CURVE
# ==================================================

pr_precision, pr_recall, _ = precision_recall_curve(
    y_test,
    test_probabilities
)

plt.figure(figsize=(7, 6))

plt.plot(
    pr_recall,
    pr_precision,
    label=f"XGBoost (AP = {pr_auc:.4f})"
)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Final XGBoost Precision-Recall Curve")
plt.legend()
plt.tight_layout()

plt.savefig(PR_PATH, dpi=300)
plt.close()


# ==================================================
# 13. COMPLETION
# ==================================================

print("\n======================================")
print("FINAL EVALUATION COMPLETED")
print("======================================")

print("Saved files:")

for path in [
    MODEL_PATH,
    METRICS_PATH,
    PREDICTIONS_PATH,
    REPORT_PATH,
    CONFUSION_PATH,
    ROC_PATH,
    PR_PATH
]:
    print(path)
