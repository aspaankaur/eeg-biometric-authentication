from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config import TABLES_DIR, FIGURES_DIR


# ---------------------------------------------------------
# Files
# ---------------------------------------------------------

SAME_FILE = TABLES_DIR / "same_condition_results.csv"
CROSS_FILE = TABLES_DIR / "cross_condition_results.csv"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Condition order
# ---------------------------------------------------------

CONDITIONS = [
    "Native_InEar",
    "NonNative_InEar",
    "Neutral_InEar",
    "Native_Bone",
    "NonNative_Bone",
    "Neutral_Bone",
]

SHORT_NAMES = {
    "Native_InEar": "Native\nIn-Ear",
    "NonNative_InEar": "Non-Native\nIn-Ear",
    "Neutral_InEar": "Neutral\nIn-Ear",
    "Native_Bone": "Native\nBone",
    "NonNative_Bone": "Non-Native\nBone",
    "Neutral_Bone": "Neutral\nBone",
}

MODELS = [
    "SVM",
    "Random Forest",
    "KNN",
]


# ---------------------------------------------------------
# Load results
# ---------------------------------------------------------

same_df = pd.read_csv(SAME_FILE)
cross_df = pd.read_csv(CROSS_FILE)

print("Same-condition rows:", len(same_df))
print("Cross-condition rows:", len(cross_df))


# ---------------------------------------------------------
# Build matrix
# ---------------------------------------------------------

def build_matrix(model_name, metric):
    matrix = pd.DataFrame(
        index=CONDITIONS,
        columns=CONDITIONS,
        dtype=float
    )

    # -----------------------------------------------------
    # Off-diagonal values:
    # train on source condition, test on target condition
    # -----------------------------------------------------

    model_cross = cross_df[
        cross_df["model"] == model_name
    ]

    for _, row in model_cross.iterrows():
        source = row["source_condition"]
        target = row["target_condition"]

        matrix.loc[source, target] = row[metric]

    # -----------------------------------------------------
    # Diagonal values:
    # use the leakage-safe same-condition experiment
    # -----------------------------------------------------

    model_same = same_df[
        same_df["model"] == model_name
    ]

    for _, row in model_same.iterrows():
        condition = row["condition"]

        matrix.loc[condition, condition] = row[metric]

    return matrix


# ---------------------------------------------------------
# Plot heatmap
# ---------------------------------------------------------

def plot_heatmap(matrix, model_name, metric):
    display_matrix = matrix.copy()

    display_matrix.index = [
        SHORT_NAMES[x]
        for x in display_matrix.index
    ]

    display_matrix.columns = [
        SHORT_NAMES[x]
        for x in display_matrix.columns
    ]

    # Convert values to percentages
    percentage_matrix = display_matrix * 100

    plt.figure(figsize=(9, 7))

    sns.heatmap(
        percentage_matrix,
        annot=True,
        fmt=".1f",
        cmap="YlGnBu",
        vmin=0,
        vmax=100,
        linewidths=0.5,
        cbar_kws={
            "label": f"{metric.upper()} (%)"
        }
    )

    metric_title = (
        "Accuracy"
        if metric == "accuracy"
        else "Macro F1-Score"
    )

    plt.title(
        f"{model_name} — {metric_title}\n"
        "Cross-Condition EEG Subject Identification",
        fontsize=13
    )

    plt.xlabel(
        "Target / Testing Condition",
        fontsize=11
    )

    plt.ylabel(
        "Source / Training Condition",
        fontsize=11
    )

    plt.xticks(
        rotation=0,
        fontsize=9
    )

    plt.yticks(
        rotation=0,
        fontsize=9
    )

    plt.tight_layout()

    safe_model_name = (
        model_name
        .lower()
        .replace(" ", "_")
    )

    output_file = (
        FIGURES_DIR /
        f"{safe_model_name}_{metric}_heatmap.png"
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {output_file}")


# ---------------------------------------------------------
# Generate matrices and figures
# ---------------------------------------------------------

for model_name in MODELS:

    print("\n-----------------------------------")
    print(f"Model: {model_name}")
    print("-----------------------------------")

    for metric in [
        "accuracy",
        "f1"
    ]:

        matrix = build_matrix(
            model_name,
            metric
        )

        print(
            f"\n{metric.upper()} matrix:"
        )

        print(
            (matrix * 100)
            .round(2)
        )

        plot_heatmap(
            matrix,
            model_name,
            metric
        )


print("\n-----------------------------------")
print("All heatmaps generated successfully.")
print("-----------------------------------")