
import pandas as pd
import joblib

from pathlib import Path
from sklearn.preprocessing import StandardScaler

# Project paths
ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data" / "raw" / "creditcard.csv"
SPLIT_PATH = ROOT / "data" / "processed" / "split_indices.csv"

OUTPUT_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load original dataset and saved split indices
df = pd.read_csv(DATA_PATH)
split_indices = pd.read_csv(SPLIT_PATH)

train_idx = split_indices.loc[
    split_indices["split"] == "train", "row_index"
].to_numpy()

test_idx = split_indices.loc[
    split_indices["split"] == "test", "row_index"
].to_numpy()

train_df = df.iloc[train_idx].copy()
test_df = df.iloc[test_idx].copy()

# 2. Separate features and target
X_train = train_df.drop(columns=["Class"])
X_test = test_df.drop(columns=["Class"])

y_train = train_df["Class"]
y_test = test_df["Class"]

# 3. Scale only Time and Amount
scale_columns = ["Time", "Amount"]

scaler = StandardScaler()

# Fit only on training data
X_train.loc[:, scale_columns] = scaler.fit_transform(
    X_train[scale_columns]
)

# Transform test data using the training scaler
X_test.loc[:, scale_columns] = scaler.transform(
    X_test[scale_columns]
)

# 4. Save processed datasets
train_processed = X_train.copy()
train_processed["Class"] = y_train.to_numpy()

test_processed = X_test.copy()
test_processed["Class"] = y_test.to_numpy()

train_processed.to_csv(
    OUTPUT_DIR / "train_processed.csv",
    index=False
)

test_processed.to_csv(
    OUTPUT_DIR / "test_processed.csv",
    index=False
)

# 5. Save fitted scaler
joblib.dump(
    scaler,
    MODEL_DIR / "feature_scaler.joblib"
)

# 6. Verification
print("\n========== PREPROCESSING SUMMARY ==========")

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

print("\nScaled features:", scale_columns)

print("\nTraining scaled feature means:")
print(X_train[scale_columns].mean())

print("\nTraining scaled feature standard deviations:")
print(X_train[scale_columns].std(ddof=0))

print("\nTest feature means:")
print(X_test[scale_columns].mean())

print("\nTraining fraud cases:", int(y_train.sum()))
print("Testing fraud cases:", int(y_test.sum()))

print("\nSaved files:")
print("train_processed.csv")
print("test_processed.csv")
print("feature_scaler.joblib")

print("\nOriginal dataset remains unchanged.")
print("========== PREPROCESSING COMPLETED ==========")
