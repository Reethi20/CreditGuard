
import pandas as pd
from pathlib import Path

# Project paths
ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "raw" / "creditcard.csv"

print("\n========== CREDITGUARD DUPLICATE INVESTIGATION ==========\n")

# Load original dataset (read-only)
df = pd.read_csv(DATA_PATH)

# --------------------------------------------------
# 1. Exact duplicate groups (including Class)
# --------------------------------------------------

print("1. EXACT DUPLICATE GROUP ANALYSIS")

full_group_counts = (
    df.groupby(list(df.columns), dropna=False)
    .size()
)

duplicate_groups = full_group_counts[full_group_counts > 1]

print("Unique exact duplicate groups:", len(duplicate_groups))
print("Rows involved in duplicate groups:", duplicate_groups.sum())
print("Excess repeated rows:", (duplicate_groups - 1).sum())

# --------------------------------------------------
# 2. Fraudulent records in duplicate groups
# --------------------------------------------------

print("\n2. FRAUD DISTRIBUTION IN DUPLICATE GROUPS")

duplicate_rows = df[df.duplicated(keep=False)]

print(duplicate_rows["Class"].value_counts().sort_index())

fraud_duplicate_groups = sum(
    group_counts > 1
    for group_counts in duplicate_groups.index
) if False else None

fraud_groups = 0
for group in duplicate_groups.index:
    if group[-1] == 1:
        fraud_groups += 1

print("Duplicate groups containing fraud:", fraud_groups)

# --------------------------------------------------
# 3. Identical features, excluding Class
# --------------------------------------------------

print("\n3. DUPLICATE FEATURES WITHOUT CLASS LABEL")

feature_columns = [col for col in df.columns if col != "Class"]

feature_groups = (
    df.groupby(feature_columns, dropna=False)
    .agg(
        record_count=("Class", "size"),
        class_count=("Class", "nunique"),
        fraud_count=("Class", "sum")
    )
)

repeated_feature_groups = feature_groups[
    feature_groups["record_count"] > 1
]

print("Repeated feature groups:", len(repeated_feature_groups))
print(
    "Rows in repeated feature groups:",
    repeated_feature_groups["record_count"].sum()
)

# --------------------------------------------------
# 4. Conflicting labels
# --------------------------------------------------

print("\n4. CONFLICTING CLASS LABELS")

conflicting_groups = repeated_feature_groups[
    repeated_feature_groups["class_count"] > 1
]

print("Groups with conflicting labels:", len(conflicting_groups))
print(
    "Rows in conflicting groups:",
    conflicting_groups["record_count"].sum()
)

if not conflicting_groups.empty:
    print("\nConflicting group examples:")
    print(conflicting_groups.head(10))

# --------------------------------------------------
# 5. Final summary
# --------------------------------------------------

print("\n========== INVESTIGATION SUMMARY ==========")

print("Original transactions:", len(df))
print("Exact duplicate groups:", len(duplicate_groups))
print("Excess repeated rows:", (duplicate_groups - 1).sum())
print("Fraudulent duplicate groups:", fraud_groups)
print("Conflicting label groups:", len(conflicting_groups))

print("\nOriginal dataset remains unchanged.")
print("\n========== INVESTIGATION COMPLETED ==========")
