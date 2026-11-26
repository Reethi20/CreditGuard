
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data" / "raw" / "creditcard.csv"
OUTPUT_DIR = ROOT / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load original data without modifying it
df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["Class"])
y = df["Class"]

# Assign identical rows the same group ID.
# All features, including Class, are considered.
groups = df.groupby(
    list(df.columns),
    sort=False,
    dropna=False
).ngroup()

# Five folds give approximately an 80:20 split.
splitter = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(X, y, groups=groups)
)

# Save indices, not transformed datasets.
split_indices = pd.DataFrame({
    "row_index": list(train_idx) + list(test_idx),
    "split": (
        ["train"] * len(train_idx)
        + ["test"] * len(test_idx)
    )
})

split_indices.to_csv(
    OUTPUT_DIR / "split_indices.csv",
    index=False
)

# Verify split
train_groups = set(groups.iloc[train_idx])
test_groups = set(groups.iloc[test_idx])

print("\n========== SPLIT SUMMARY ==========")

print("Original records:", len(df))
print("Training records:", len(train_idx))
print("Testing records:", len(test_idx))

print("\nTraining class distribution:")
print(y.iloc[train_idx].value_counts())

print("\nTesting class distribution:")
print(y.iloc[test_idx].value_counts())

print("\nTraining fraud percentage:")
print(round(y.iloc[train_idx].mean() * 100, 4), "%")

print("\nTesting fraud percentage:")
print(round(y.iloc[test_idx].mean() * 100, 4), "%")

print("\nDuplicate group overlap:", len(train_groups & test_groups))

print("\nOriginal dataset unchanged.")
print("Split indices saved successfully.")
