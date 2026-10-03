from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[1]

RESULT_FILE = (
    ROOT
    / "results"
    / "tables"
    / "controlled_enrollment_results.csv"
)

OUTPUT_FILE = (
    ROOT
    / "results"
    / "tables"
    / "statistical_analysis.csv"
)

RANDOM_STATE = 42
N_BOOTSTRAP = 10000


# ============================================================
# LOAD RESULTS
# ============================================================

df = pd.read_csv(RESULT_FILE)

print("\n========================================")
print("STATISTICAL ANALYSIS")
print("========================================")

print("Rows:", len(df))


# ============================================================
# TARGET-LEVEL MEANS
# ============================================================

target_means = (
    df.groupby(
        ["target_condition", "enrollment_size"]
    )["accuracy"]
    .mean()
    .reset_index()
)

pivot = target_means.pivot(
    index="target_condition",
    columns="enrollment_size",
    values="accuracy",
)

print("\n========================================")
print("TARGET-LEVEL MEAN ACCURACY")
print("========================================")

print((pivot * 100).round(2))


# ============================================================
# 1 CONDITION VS 5 CONDITIONS
# ============================================================

one_condition = pivot[1].to_numpy()
five_condition = pivot[5].to_numpy()

differences = five_condition - one_condition

mean_difference = differences.mean()

print("\n========================================")
print("PAIRED IMPROVEMENTS")
print("========================================")

for target, one, five, diff in zip(
    pivot.index,
    one_condition,
    five_condition,
    differences,
):
    print(
        f"{target:20s}: "
        f"{one * 100:.2f}% -> "
        f"{five * 100:.2f}% "
        f"({diff * 100:+.2f} pp)"
    )

print(
    f"\nMean improvement: "
    f"{mean_difference * 100:.2f} percentage points"
)


# ============================================================
# BOOTSTRAP CI OF MEAN TARGET-LEVEL IMPROVEMENT
# ============================================================

rng = np.random.default_rng(RANDOM_STATE)

bootstrap_means = []

for _ in range(N_BOOTSTRAP):

    sampled_differences = rng.choice(
        differences,
        size=len(differences),
        replace=True,
    )

    bootstrap_means.append(
        sampled_differences.mean()
    )

lower = np.percentile(
    bootstrap_means,
    2.5,
)

upper = np.percentile(
    bootstrap_means,
    97.5,
)

print("\n========================================")
print("BOOTSTRAP 95% CONFIDENCE INTERVAL")
print("========================================")

print(
    f"Mean improvement: "
    f"{mean_difference * 100:.2f} pp"
)

print(
    f"95% bootstrap CI: "
    f"[{lower * 100:.2f}, "
    f"{upper * 100:.2f}] pp"
)


# ============================================================
# WILCOXON SIGNED-RANK TEST
# ============================================================

statistic, p_value = wilcoxon(
    five_condition,
    one_condition,
    alternative="greater",
)

print("\n========================================")
print("WILCOXON SIGNED-RANK TEST")
print("========================================")

print(
    f"Statistic: {statistic:.4f}"
)

print(
    f"One-sided p-value: {p_value:.6f}"
)


# ============================================================
# SIMPLE EFFECT SIZE
# ============================================================

mean_diff = differences.mean()
sd_diff = differences.std(ddof=1)

cohens_dz = (
    mean_diff / sd_diff
    if sd_diff != 0
    else np.inf
)

print("\n========================================")
print("PAIRED EFFECT SIZE")
print("========================================")

print(
    f"Cohen's dz: {cohens_dz:.3f}"
)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary = pd.DataFrame(
    {
        "comparison": [
            "5-condition vs 1-condition"
        ],
        "mean_improvement_pp": [
            mean_difference * 100
        ],
        "ci_lower_pp": [
            lower * 100
        ],
        "ci_upper_pp": [
            upper * 100
        ],
        "wilcoxon_statistic": [
            statistic
        ],
        "wilcoxon_p_value": [
            p_value
        ],
        "cohens_dz": [
            cohens_dz
        ],
    }
)

summary.to_csv(
    OUTPUT_FILE,
    index=False,
)

print("\nSaved:", OUTPUT_FILE)