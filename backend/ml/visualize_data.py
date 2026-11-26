
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data/raw/creditcard.csv"

OUTPUT_DIR = ROOT / "docs/figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

sns.set_theme(style="whitegrid")

# 1. Class distribution
plt.figure(figsize=(8, 5))

class_counts = df["Class"].value_counts().sort_index()

sns.barplot(
    x=["Legitimate", "Fraudulent"],
    y=class_counts.values
)

plt.title("Credit Card Transaction Class Distribution")
plt.ylabel("Number of Transactions")
plt.xlabel("Transaction Class")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "class_distribution.png", dpi=300)
plt.show()

# 2. Transaction amount distribution
plt.figure(figsize=(9, 5))

sns.histplot(
    data=df,
    x="Amount",
    bins=100,
    kde=True
)

plt.title("Transaction Amount Distribution")
plt.xlabel("Transaction Amount")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "amount_distribution.png", dpi=300)
plt.show()

# 3. Fraud vs legitimate amount
plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="Class",
    y="Amount"
)

plt.title("Transaction Amount by Class")
plt.xticks([0, 1], ["Legitimate", "Fraudulent"])
plt.xlabel("Transaction Class")
plt.ylabel("Transaction Amount")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "amount_by_class.png", dpi=300)
plt.show()

# 4. Correlation heatmap
plt.figure(figsize=(16, 12))

correlation = df.corr()

sns.heatmap(
    correlation,
    cmap="coolwarm",
    center=0
)

plt.title("Credit Card Transaction Feature Correlation")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=300)
plt.show()

print("All visualisations generated successfully!")
print("Saved in:", OUTPUT_DIR)
