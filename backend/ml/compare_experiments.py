
from pathlib import Path

import pandas as pd


# --------------------------------------------------
# 1. Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = PROJECT_ROOT / "models" / "experiments"

OUTPUT_PATH = RESULTS_DIR / "master_comparison.csv"


# --------------------------------------------------
# 2. Column name mappings
# --------------------------------------------------

COLUMN_ALIASES = {
    "Model": ["Model", "experiment"],
    "Accuracy": ["Accuracy", "accuracy"],
    "Precision": ["Precision", "precision"],
    "Recall": ["Recall", "recall"],
    "F1": ["F1", "f1_score"],
    "PR_AUC": ["PR_AUC", "pr_auc"],
    "ROC_AUC": ["ROC_AUC", "roc_auc"],
    "MCC": ["MCC", "mcc"],
    "Specificity": ["Specificity", "specificity"],
    "TP": ["TP", "true_positives"],
    "FP": ["FP", "false_positives"],
    "TN": ["TN", "true_negatives"],
    "FN": ["FN", "false_negatives"],
    "Training_Time_Seconds": [
        "Training_Time_Seconds",
        "training_time_seconds"
    ],
}


# --------------------------------------------------
# 3. Read and standardise one experiment file
# --------------------------------------------------

def standardise_results(file_path):

    df = pd.read_csv(file_path)

    if df.empty:
        print(f"Skipping empty file: {file_path.name}")
        return None

    standardised = pd.DataFrame(index=df.index)

    for standard_name, possible_names in COLUMN_ALIASES.items():

        found_column = next(
            (
                column
                for column in possible_names
                if column in df.columns
            ),
            None
        )

        if found_column is not None:
            standardised[standard_name] = df[found_column]

    # Every result needs a model/configuration name.
    if "Model" not in standardised.columns:
        raise ValueError(
            f"No model name column found in {file_path.name}"
        )

    # Confusion matrix values are required to calculate
    # missing accuracy.
    confusion_columns = ["TP", "FP", "TN", "FN"]

    missing_confusion = [
        column
        for column in confusion_columns
        if column not in standardised.columns
    ]

    if missing_confusion:
        raise ValueError(
            f"{file_path.name} is missing confusion matrix "
            f"columns: {missing_confusion}"
        )

    for column in confusion_columns:
        standardised[column] = pd.to_numeric(
            standardised[column],
            errors="raise"
        )

    standardised["Total_Transactions"] = (
        standardised["TP"]
        + standardised["FP"]
        + standardised["TN"]
        + standardised["FN"]
    )

    # All experiments should use the same validation set.
    invalid_totals = (
        standardised["Total_Transactions"] != 45569
    )

    if invalid_totals.any():
        raise ValueError(
            f"Unexpected validation sample count in "
            f"{file_path.name}. Expected 45569."
        )

    # Calculate accuracy only when it wasn't saved.
    if "Accuracy" not in standardised.columns:

        standardised["Accuracy"] = (
            standardised["TP"] + standardised["TN"]
        ) / standardised["Total_Transactions"]

        standardised["Accuracy_Source"] = (
            "Calculated from confusion matrix"
        )

    else:

        standardised["Accuracy"] = pd.to_numeric(
            standardised["Accuracy"],
            errors="raise"
        )

        standardised["Accuracy_Source"] = "Saved by experiment"

    # Record the source file.
    standardised["Source_File"] = file_path.name

    return standardised


# --------------------------------------------------
# 4. Load all experiment results
# --------------------------------------------------

def load_all_experiments():

    print("Loading experiment results...")

    result_files = sorted(
        RESULTS_DIR.glob("*_results.csv")
    )

    # Keep threshold experiments separate.
    result_files = [
        file
        for file in result_files
        if "threshold" not in file.name.lower()
    ]

    if not result_files:
        raise FileNotFoundError(
            "No experiment result files found."
        )

    all_results = []

    for file_path in result_files:

        print(f"Reading: {file_path.name}")

        result = standardise_results(file_path)

        if result is not None:
            all_results.append(result)

    comparison = pd.concat(
        all_results,
        ignore_index=True
    )

    return comparison


# --------------------------------------------------
# 5. Rank models
# --------------------------------------------------

def rank_models(comparison):

    # Higher values are better for both metrics.
    comparison["F1_Rank"] = (
        comparison["F1"]
        .rank(method="min", ascending=False)
        .astype(int)
    )

    comparison["PR_AUC_Rank"] = (
        comparison["PR_AUC"]
        .rank(method="min", ascending=False)
        .astype(int)
    )

    comparison = comparison.sort_values(
        by="F1",
        ascending=False
    ).reset_index(drop=True)

    return comparison


# --------------------------------------------------
# 6. Display leaderboard
# --------------------------------------------------

def display_leaderboard(comparison):

    display_columns = [
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "PR_AUC",
        "ROC_AUC",
        "MCC",
        "Specificity",
        "TP",
        "FP",
        "TN",
        "FN",
        "Training_Time_Seconds",
        "F1_Rank",
        "PR_AUC_Rank",
    ]

    print("\n")
    print("=" * 110)
    print("CREDITGUARD - MASTER MODEL COMPARISON")
    print("=" * 110)

    print(
        comparison[display_columns].to_string(
            index=False,
            formatters={
                "Accuracy": "{:.4%}".format,
                "Precision": "{:.4f}".format,
                "Recall": "{:.4f}".format,
                "F1": "{:.4f}".format,
                "PR_AUC": "{:.4f}".format,
                "ROC_AUC": "{:.4f}".format,
                "MCC": "{:.4f}".format,
                "Specificity": "{:.4f}".format,
                "Training_Time_Seconds": "{:.2f}".format,
            }
        )
    )

    print("\nTotal model configurations:", len(comparison))

    print("\nAccuracy sources:")
    print(comparison["Accuracy_Source"].value_counts())


# --------------------------------------------------
# 7. Save master comparison
# --------------------------------------------------

def save_comparison(comparison):

    comparison.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\nMaster comparison saved successfully:")
    print(OUTPUT_PATH)


# --------------------------------------------------
# 8. Main
# --------------------------------------------------

def main():

    comparison = load_all_experiments()

    comparison = rank_models(comparison)

    display_leaderboard(comparison)

    save_comparison(comparison)

    print("\nCOMPARISON COMPLETED.")


if __name__ == "__main__":
    main()
