
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data/raw/creditcard.csv"
SPLIT_PATH = ROOT / "data/processed/experiment_splits.csv"
MODEL_PATH = ROOT / "models/final/final_xgboost_model.joblib"

OUTPUT_DIR = ROOT / "models/experiments/shap"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. LOAD DATA AND FINAL MODEL
# --------------------------------------------------

print("Loading original dataset...")

data = pd.read_csv(DATA_PATH)
splits = pd.read_csv(SPLIT_PATH)

model = joblib.load(MODEL_PATH)

feature_columns = [col for col in data.columns if col != "Class"]

# Use development data only (training + validation).
development_indices = splits.loc[
    splits["split"].isin(["train", "validation"]),
    "row_index"
].astype(int)

development_data = data.iloc[development_indices.to_numpy()].copy()

print(f"Development rows: {len(development_data)}")
print(f"Features: {len(feature_columns)}")

# Confirm that the model expects the same feature structure.
X_development = development_data[feature_columns]
y_development = development_data["Class"]


# --------------------------------------------------
# 3. SELECT AN EXPLANATION SAMPLE
# --------------------------------------------------

# Select 200 legitimate and 200 fraudulent development
# transactions to make both classes visible in explanations.
legitimate = development_data[
    development_data["Class"] == 0
].sample(n=200, random_state=42)

fraudulent = development_data[
    development_data["Class"] == 1
].sample(n=200, random_state=42)

explanation_data = pd.concat(
    [legitimate, fraudulent]
).sample(frac=1, random_state=42)

X_explain = explanation_data[feature_columns]

# Use legitimate development transactions as the reference
# background for probability-based SHAP explanations.
background = development_data[
    development_data["Class"] == 0
].sample(n=100, random_state=42)[feature_columns]

print(f"Explanation sample: {len(X_explain)} rows")
print(f"Background sample: {len(background)} rows")


# --------------------------------------------------
# 4. CREATE SHAP EXPLAINER
# --------------------------------------------------

print("\nCreating SHAP explainer...")

start_time = time.time()

explainer = shap.TreeExplainer(
    model,
    data=background,
    model_output="probability",
    feature_perturbation="interventional"
)

shap_values = explainer(X_explain)

print(
    "SHAP calculations completed in "
    f"{time.time() - start_time:.2f} seconds."
)


# --------------------------------------------------
# 5. GLOBAL FEATURE IMPORTANCE
# --------------------------------------------------

print("\nGenerating global feature importance...")

# Beeswarm summary plot
plt.figure()
shap.plots.beeswarm(shap_values, max_display=20, show=False)
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "shap_summary.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# Mean absolute SHAP importance
plt.figure()
shap.plots.bar(shap_values, max_display=20, show=False)
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "shap_feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()


# --------------------------------------------------
# 6. EXPLAIN ONE FRAUD PREDICTION
# --------------------------------------------------

print("\nGenerating individual transaction explanation...")

probabilities = model.predict_proba(X_explain)[:, 1]

# Select the fraudulent transaction with the highest
# predicted fraud probability in the explanation sample.
fraud_positions = np.where(
    explanation_data["Class"].to_numpy() == 1
)[0]

selected_position = fraud_positions[
    np.argmax(probabilities[fraud_positions])
]

selected_probability = probabilities[selected_position]

print(f"Selected transaction position: {selected_position}")
print(f"Actual class: Fraud")
print(f"Predicted fraud probability: {selected_probability:.6f}")

# Waterfall explanation
plt.figure()
shap.plots.waterfall(
    shap_values[selected_position],
    max_display=15,
    show=False
)
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "shap_waterfall.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()


# --------------------------------------------------
# 7. SAVE NUMERICAL SHAP IMPORTANCE
# --------------------------------------------------

importance = pd.DataFrame({
    "Feature": feature_columns,
    "Mean_Absolute_SHAP": np.abs(
        shap_values.values
    ).mean(axis=0)
})

importance = importance.sort_values(
    "Mean_Absolute_SHAP",
    ascending=False
)

importance.to_csv(
    OUTPUT_DIR / "shap_importance.csv",
    index=False
)

print("\nTop 10 influential features:")
print(importance.head(10).to_string(index=False))


# --------------------------------------------------
# 8. COMPLETION
# --------------------------------------------------

print("\n======================================")
print("SHAP EXPLAINABILITY COMPLETED")
print("======================================")

print(f"Output directory: {OUTPUT_DIR}")
print("Generated files:")
print("- shap_summary.png")
print("- shap_feature_importance.png")
print("- shap_waterfall.png")
print("- shap_importance.csv")
