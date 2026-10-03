"""
Data-volume-controlled multi-condition enrollment experiment.

Purpose:
Test whether multi-condition enrollment improves unseen-condition
EEG identification when the TOTAL amount of training data is held constant.

For every target condition:
- target condition is completely unseen during training
- use 580 total training windows for every enrollment size
- distribute those windows as evenly as possible across selected conditions
- repeat random sampling to reduce dependence on one subsample
"""

from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score


# ============================================================
# SETTINGS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

FEATURE_FILE = (
    ROOT / "results" / "tables" / "eeg_features.csv"
)

RESULT_FILE = (
    ROOT / "results" / "tables"
    / "controlled_enrollment_results.csv"
)

SUMMARY_FILE = (
    ROOT / "results" / "tables"
    / "controlled_enrollment_summary.csv"
)

TARGET_SUMMARY_FILE = (
    ROOT / "results" / "tables"
    / "controlled_enrollment_target_summary.csv"
)

CONDITIONS = [
    "Native_InEar",
    "NonNative_InEar",
    "Neutral_InEar",
    "Native_Bone",
    "NonNative_Bone",
    "Neutral_Bone",
]

ENROLLMENT_SIZES = [1, 2, 3, 5]

# Same total amount of EEG used for every training configuration.
# One complete condition = 20 subjects × 29 windows = 580 windows.
TRAINING_BUDGET = 580

# Repeated subsampling for more stable estimates.
N_REPEATS = 10

RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

print("\n========================================")
print("DATA-VOLUME-CONTROLLED ENROLLMENT")
print("========================================")

df = pd.read_csv(FEATURE_FILE)

print("Rows:", len(df))
print("Columns:", len(df.columns))
print("Subjects:", df["subject"].nunique())
print("Conditions:", df["condition"].nunique())


# ============================================================
# FEATURES
# ============================================================

NON_FEATURE_COLUMNS = {
    "subject",
    "experiment",
    "condition",
    "recording",
    "window",
    "window_index",
    "window_number",
    "window_start_sample",
    "transducer",
}

feature_columns = [
    column
    for column in df.columns
    if column not in NON_FEATURE_COLUMNS
    and pd.api.types.is_numeric_dtype(df[column])
]

print("Feature columns:", len(feature_columns))

if len(feature_columns) != 64:
    raise ValueError(
        f"Expected 64 EEG features, found {len(feature_columns)}."
    )


# ============================================================
# BALANCED SAMPLING
# ============================================================

def sample_training_data(
    source_df,
    train_conditions,
    total_budget,
    random_state,
):
    """
    Sample approximately equal numbers of windows from every
    subject-condition combination.

    This prevents a condition or subject from dominating the
    controlled training set.
    """

    rng = np.random.default_rng(random_state)

    groups = []

    # Number of subject-condition groups.
    n_groups = (
        len(train_conditions)
        * source_df["subject"].nunique()
    )

    base_samples = total_budget // n_groups
    remainder = total_budget % n_groups

    group_info = []

    for condition in train_conditions:
        for subject in sorted(source_df["subject"].unique()):

            group = source_df[
                (source_df["condition"] == condition)
                & (source_df["subject"] == subject)
            ]

            if len(group) == 0:
                raise ValueError(
                    f"No data for subject {subject}, "
                    f"condition {condition}"
                )

            group_info.append(
                (condition, subject, group)
            )

    # Randomly decide which groups receive an extra sample
    # when the budget is not exactly divisible.
    extra_indices = set()

    if remainder > 0:
        extra_indices = set(
            rng.choice(
                len(group_info),
                size=remainder,
                replace=False,
            )
        )

    for index, (_, _, group) in enumerate(group_info):

        n_samples = base_samples

        if index in extra_indices:
            n_samples += 1

        if n_samples > len(group):
            raise ValueError(
                "Training budget requires more windows "
                "than are available in a subject-condition group."
            )

        sampled = group.sample(
            n=n_samples,
            replace=False,
            random_state=(
                random_state * 1000 + index
            ),
        )

        groups.append(sampled)

    sampled_df = pd.concat(
        groups,
        ignore_index=True,
    )

    return sampled_df


# ============================================================
# EXPERIMENT
# ============================================================

results = []

for target_condition in CONDITIONS:

    print("\n========================================")
    print("UNSEEN TARGET:", target_condition)
    print("========================================")

    test_df = df[
        df["condition"] == target_condition
    ].copy()

    available_conditions = [
        condition
        for condition in CONDITIONS
        if condition != target_condition
    ]

    X_test = test_df[feature_columns]
    y_test = test_df["subject"]

    for enrollment_size in ENROLLMENT_SIZES:

        configurations = list(
            combinations(
                available_conditions,
                enrollment_size,
            )
        )

        print(
            f"\nEnrollment size {enrollment_size}: "
            f"{len(configurations)} configurations"
        )

        for config_index, train_conditions in enumerate(
            configurations,
            start=1,
        ):

            candidate_train_df = df[
                df["condition"].isin(train_conditions)
            ].copy()

            for repeat in range(N_REPEATS):

                seed = (
                    RANDOM_STATE
                    + repeat
                    + config_index * 100
                    + CONDITIONS.index(target_condition) * 10000
                    + enrollment_size * 100000
                )

                train_df = sample_training_data(
                    candidate_train_df,
                    train_conditions,
                    TRAINING_BUDGET,
                    seed,
                )

                if len(train_df) != TRAINING_BUDGET:
                    raise ValueError(
                        "Controlled training budget was not preserved."
                    )

                X_train = train_df[feature_columns]
                y_train = train_df["subject"]

                model = RandomForestClassifier(
                    n_estimators=200,
                    random_state=seed,
                    n_jobs=-1,
                )

                model.fit(
                    X_train,
                    y_train,
                )

                predictions = model.predict(
                    X_test
                )

                accuracy = accuracy_score(
                    y_test,
                    predictions,
                )

                f1 = f1_score(
                    y_test,
                    predictions,
                    average="macro",
                    zero_division=0,
                )

                results.append(
                    {
                        "target_condition":
                            target_condition,

                        "enrollment_size":
                            enrollment_size,

                        "train_conditions":
                            "|".join(train_conditions),

                        "repeat":
                            repeat + 1,

                        "training_windows":
                            len(train_df),

                        "test_windows":
                            len(test_df),

                        "accuracy":
                            accuracy,

                        "f1":
                            f1,
                    }
                )


# ============================================================
# SAVE RAW RESULTS
# ============================================================

results_df = pd.DataFrame(results)

RESULT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

results_df.to_csv(
    RESULT_FILE,
    index=False,
)


# ============================================================
# OVERALL SUMMARY
# ============================================================

summary = (
    results_df
    .groupby("enrollment_size")
    .agg(
        experiments=("accuracy", "count"),
        mean_accuracy=("accuracy", "mean"),
        std_accuracy=("accuracy", "std"),
        mean_f1=("f1", "mean"),
        std_f1=("f1", "std"),
    )
    .reset_index()
)

summary.to_csv(
    SUMMARY_FILE,
    index=False,
)


print("\n========================================")
print("CONTROLLED OVERALL PERFORMANCE")
print("========================================")

display_summary = summary.copy()

for column in [
    "mean_accuracy",
    "std_accuracy",
    "mean_f1",
    "std_f1",
]:
    display_summary[column] *= 100

print(
    display_summary
    .round(2)
    .to_string(index=False)
)


# ============================================================
# TARGET-WISE SUMMARY
# ============================================================

target_summary = (
    results_df
    .groupby(
        [
            "enrollment_size",
            "target_condition",
        ]
    )
    .agg(
        mean_accuracy=("accuracy", "mean"),
        std_accuracy=("accuracy", "std"),
        mean_f1=("f1", "mean"),
        experiments=("accuracy", "count"),
    )
    .reset_index()
)

target_summary.to_csv(
    TARGET_SUMMARY_FILE,
    index=False,
)


print("\n========================================")
print("CONTROLLED TARGET-WISE PERFORMANCE")
print("========================================")

target_display = target_summary.copy()

for column in [
    "mean_accuracy",
    "std_accuracy",
    "mean_f1",
]:
    target_display[column] *= 100

print(
    target_display
    .round(2)
    .to_string(index=False)
)


# ============================================================
# IMPROVEMENT OVER SINGLE-CONDITION
# ============================================================

baseline = summary.loc[
    summary["enrollment_size"] == 1,
    "mean_accuracy",
].iloc[0]

print("\n========================================")
print("GAIN OVER CONTROLLED 1-CONDITION BASELINE")
print("========================================")

for _, row in summary.iterrows():

    size = int(
        row["enrollment_size"]
    )

    gain = (
        row["mean_accuracy"]
        - baseline
    ) * 100

    print(
        f"{size} condition(s): "
        f"{row['mean_accuracy'] * 100:.2f}% "
        f"(gain: {gain:+.2f} pp)"
    )


print("\n========================================")
print("EXPERIMENT COMPLETE")
print("========================================")

print("Training budget per model:", TRAINING_BUDGET)
print("Repeats per configuration:", N_REPEATS)

print("\nSaved:")
print(RESULT_FILE)
print(SUMMARY_FILE)
print(TARGET_SUMMARY_FILE)