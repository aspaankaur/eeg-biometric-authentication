import pandas as pd

from config import TABLES_DIR


# ---------------------------------------------------------
# Load results
# ---------------------------------------------------------

same_df = pd.read_csv(
    TABLES_DIR / "same_condition_results.csv"
)

cross_df = pd.read_csv(
    TABLES_DIR / "cross_condition_results.csv"
)


# ---------------------------------------------------------
# Helper: identify transducer type
# ---------------------------------------------------------

def get_transducer(condition):
    if "InEar" in condition:
        return "InEar"
    elif "Bone" in condition:
        return "Bone"
    return "Unknown"


cross_df["source_transducer"] = (
    cross_df["source_condition"].apply(get_transducer)
)

cross_df["target_transducer"] = (
    cross_df["target_condition"].apply(get_transducer)
)

cross_df["transducer_change"] = (
    cross_df["source_transducer"]
    != cross_df["target_transducer"]
)


# ---------------------------------------------------------
# 1. Average same-condition performance
# ---------------------------------------------------------

print("\n========================================")
print("1. AVERAGE SAME-CONDITION PERFORMANCE")
print("========================================")

same_summary = (
    same_df
    .groupby("model")[["accuracy", "f1"]]
    .mean()
    .sort_values("accuracy", ascending=False)
    * 100
)

print(same_summary.round(2))


# ---------------------------------------------------------
# 2. Average cross-condition performance
# ---------------------------------------------------------

print("\n========================================")
print("2. AVERAGE CROSS-CONDITION PERFORMANCE")
print("========================================")

cross_summary = (
    cross_df
    .groupby("model")[["accuracy", "f1"]]
    .mean()
    .sort_values("accuracy", ascending=False)
    * 100
)

print(cross_summary.round(2))


# ---------------------------------------------------------
# 3. Performance drop
# ---------------------------------------------------------

print("\n========================================")
print("3. SAME → CROSS PERFORMANCE DROP")
print("========================================")

comparison = pd.DataFrame({
    "same_accuracy":
        same_df.groupby("model")["accuracy"].mean(),

    "cross_accuracy":
        cross_df.groupby("model")["accuracy"].mean(),

    "same_f1":
        same_df.groupby("model")["f1"].mean(),

    "cross_f1":
        cross_df.groupby("model")["f1"].mean(),
})

comparison["accuracy_drop"] = (
    comparison["same_accuracy"]
    - comparison["cross_accuracy"]
)

comparison["f1_drop"] = (
    comparison["same_f1"]
    - comparison["cross_f1"]
)

print((comparison * 100).round(2))


# ---------------------------------------------------------
# 4. Best and worst cross-condition pair per model
# ---------------------------------------------------------

print("\n========================================")
print("4. BEST / WORST CROSS-CONDITION PAIRS")
print("========================================")

for model in cross_df["model"].unique():

    model_df = cross_df[
        cross_df["model"] == model
    ]

    best = model_df.loc[
        model_df["accuracy"].idxmax()
    ]

    worst = model_df.loc[
        model_df["accuracy"].idxmin()
    ]

    print(f"\n{model}")

    print(
        "BEST:",
        best["source_condition"],
        "->",
        best["target_condition"],
        f"| Accuracy: {best['accuracy'] * 100:.2f}%",
        f"| F1: {best['f1'] * 100:.2f}%"
    )

    print(
        "WORST:",
        worst["source_condition"],
        "->",
        worst["target_condition"],
        f"| Accuracy: {worst['accuracy'] * 100:.2f}%",
        f"| F1: {worst['f1'] * 100:.2f}%"
    )


# ---------------------------------------------------------
# 5. Same-transducer vs cross-transducer
# ---------------------------------------------------------

print("\n========================================")
print("5. TRANSDUCER ROBUSTNESS")
print("========================================")

transducer_summary = (
    cross_df
    .groupby(
        ["model", "transducer_change"]
    )[["accuracy", "f1"]]
    .mean()
    * 100
)

transducer_summary = (
    transducer_summary
    .rename(index={
        False: "Same transducer",
        True: "Cross transducer"
    })
)

print(transducer_summary.round(2))


# ---------------------------------------------------------
# 6. Average performance by source condition
# ---------------------------------------------------------

print("\n========================================")
print("6. SOURCE CONDITION PERFORMANCE")
print("========================================")

source_summary = (
    cross_df
    .groupby(
        ["model", "source_condition"]
    )[["accuracy", "f1"]]
    .mean()
    * 100
)

print(source_summary.round(2))


# ---------------------------------------------------------
# 7. Average performance by target condition
# ---------------------------------------------------------

print("\n========================================")
print("7. TARGET CONDITION PERFORMANCE")
print("========================================")

target_summary = (
    cross_df
    .groupby(
        ["model", "target_condition"]
    )[["accuracy", "f1"]]
    .mean()
    * 100
)

print(target_summary.round(2))


# ---------------------------------------------------------
# Save summary tables
# ---------------------------------------------------------

same_summary.to_csv(
    TABLES_DIR / "summary_same_condition.csv"
)

cross_summary.to_csv(
    TABLES_DIR / "summary_cross_condition.csv"
)

(comparison * 100).to_csv(
    TABLES_DIR / "summary_performance_drop.csv"
)

transducer_summary.to_csv(
    TABLES_DIR / "summary_transducer.csv"
)

source_summary.to_csv(
    TABLES_DIR / "summary_source_condition.csv"
)

target_summary.to_csv(
    TABLES_DIR / "summary_target_condition.csv"
)


print("\n========================================")
print("Analysis complete.")
print("Summary tables saved in results/tables/")
print("========================================")
