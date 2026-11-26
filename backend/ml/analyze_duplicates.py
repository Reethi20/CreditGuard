
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
df = pd.read_csv(ROOT / "data/raw/creditcard.csv")

# Identify duplicated rows
duplicate_mask = df.duplicated(keep=False)
duplicates = df[duplicate_mask]

print("\n========== DUPLICATE ANALYSIS ==========")

print("Rows involved in duplicate groups:", len(duplicates))

print("\nClass distribution among duplicate rows:")
print(duplicates["Class"].value_counts().sort_index())

print("\nDuplicate group statistics:")
print(
    duplicates.groupby("Class")
    .size()
    .rename(index={0: "Legitimate", 1: "Fraudulent"})
)

print("\nFraud distribution in full dataset:")
print(df["Class"].value_counts(normalize=True) * 100)

print("\nFraud distribution among duplicate rows:")
print(duplicates["Class"].value_counts(normalize=True) * 100)

print("\n========== ANALYSIS COMPLETED ==========")
