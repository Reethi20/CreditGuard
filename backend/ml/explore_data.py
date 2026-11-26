
import pandas as pd
from pathlib import Path

# Locate the dataset
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"

# Load dataset
print("\n========== CREDITGUARD DATA EXPLORATION ==========\n")

df = pd.read_csv(DATA_PATH)

# 1. Dataset dimensions
print("1. DATASET SHAPE")
print(f"Total transactions: {df.shape[0]:,}")
print(f"Total columns: {df.shape[1]}")

# 2. First five records
print("\n2. FIRST FIVE TRANSACTIONS")
print(df.head())

# 3. Column information
print("\n3. DATASET INFORMATION")
df.info()

# 4. Missing values
print("\n4. MISSING VALUES")
print(df.isnull().sum())

print(f"\nTotal missing values: {df.isnull().sum().sum()}")

# 5. Duplicate records
print("\n5. DUPLICATE RECORDS")
print(f"Duplicate rows: {df.duplicated().sum():,}")

# 6. Fraud distribution
print("\n6. FRAUD DISTRIBUTION")

class_counts = df["Class"].value_counts().sort_index()

for label, count in class_counts.items():
    name = "Fraudulent" if label == 1 else "Legitimate"
    percentage = (count / len(df)) * 100

    print(f"{name}: {count:,} ({percentage:.4f}%)")

# 7. Transaction amount statistics
print("\n7. TRANSACTION AMOUNT STATISTICS")
print(df["Amount"].describe())

# 8. Fraud vs legitimate amount
print("\n8. AMOUNT BY TRANSACTION CLASS")
print(df.groupby("Class")["Amount"].agg(
    ["count", "mean", "median", "min", "max"]
))

print("\n========== EXPLORATION COMPLETED ==========")
