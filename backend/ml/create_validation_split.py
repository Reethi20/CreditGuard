
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data/raw/creditcard.csv"
SPLIT_PATH = ROOT / "data/processed/split_indices.csv"
OUTPUT_PATH = ROOT / "data/processed/experiment_splits.csv"

# Load original dataset
df = pd.read_csv(DATA_PATH)

# Load our existing outer train/test split
outer_split = pd.read_csv(SPLIT_PATH)

development_idx = outer_split.loc[
    outer_split["split"] == "train", "row_index"
].to_numpy()

test_idx = outer_split.loc[
    outer_split["split"] == "test", "row_index"
].to_numpy()

# Recover duplicate group IDs from original records
groups = df.groupby(
    list(df.columns),
    sort=False,
    dropna=False
).ngroup()

# Development dataset only
dev_df = df.iloc[development_idx]

dev_groups = groups.iloc[development_idx]

# Create group-aware, stratified validation split
splitter = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

train_local, val_local = next(
    splitter.split(
        dev_df.drop(columns=["Class"]),
        dev_df["Class"],
        groups=dev_groups
    )
)

train_idx = development_idx[train_local]
val_idx = development_idx[val_local]

# Verify duplicate group separation
train_groups = set(groups.iloc[train_idx])
val_groups = set(groups.iloc[val_idx])
test_groups = set(groups.iloc[test_idx])

assert not (train_groups & val_groups)
assert not (train_groups & test_groups)
assert not (val_groups & test_groups)

# Save original row indices and their partitions
result = pd.DataFrame({
    "row_index": (
        list(train_idx) +
        list(val_idx) +
        list(test_idx)
    ),
    "split": (
        ["train"] * len(train_idx) +
        ["validation"] * len(val_idx) +
        ["test"] * len(test_idx)
    )
})

result.to_csv(OUTPUT_PATH, index=False)

print("\n========== EXPERIMENTAL SPLIT SUMMARY ==========")

for name, indices in [
    ("Training", train_idx),
    ("Validation", val_idx),
    ("Testing", test_idx)
]:
    subset = df.iloc[indices]
    print(f"\n{name} records: {len(subset):,}")
    print(f"Fraud cases: {subset['Class'].sum():,}")
    print(
        "Fraud percentage:",
        round(subset["Class"].mean() * 100, 4),
        "%"
    )

print("\nDuplicate group overlap:")
print("Train vs Validation:", len(train_groups & val_groups))
print("Train vs Test:", len(train_groups & test_groups))
print("Validation vs Test:", len(val_groups & test_groups))

print("\nSaved:", OUTPUT_PATH)
print("Original dataset remains unchanged.")
print("========== COMPLETED ==========")
