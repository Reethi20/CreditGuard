
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

df = pd.read_csv(ROOT / "data/raw/creditcard.csv")

output_dir = ROOT / "docs/figures"
output_dir.mkdir(parents=True, exist_ok=True)

# Log transformation for visualisation only
df["LogAmount"] = np.log1p(df["Amount"])

plt.figure(figsize=(10, 6))

for label, name in [(0, "Legitimate"), (1, "Fraudulent")]:
    values = df.loc[df["Class"] == label, "LogAmount"]
    plt.hist(values, bins=60, alpha=0.55, label=name)

plt.title("Log-Transformed Transaction Amount Distribution")
plt.xlabel("log(1 + Transaction Amount)")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()

plt.savefig(
    output_dir / "log_amount_distribution.png",
    dpi=300
)

plt.close()

print("Logarithmic amount visualisation saved!")
