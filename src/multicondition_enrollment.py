"""
Multi-condition enrollment experiment for EEG biometric identification.

Paper 2 experiment:
Compare identification performance when training on 1, 2, 3, or 5
auditory conditions and testing on a completely unseen target condition.

The target condition is never used during training.
"""

from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

FEATURE_FILE = ROOT / "results" / "tables" / "eeg_features.csv"

OUTPUT_FILE = (
    ROOT
    / "results"
    / "tables"
    / "multicondition_enrollment_results.csv"
)

SUMMARY_FILE = (
    ROOT
    / "results"
    / "tables"
    / "multicondition_enrollment_summary.csv"
)


# ============================================================
# EXPERIMENT SETTINGS
# ============================================================

CONDITIONS = [
    "Native_InEar",
    "NonNative_InEar",
    "Neutral_InEar",
    "Native_Bone",
    "NonNative_Bone",
    "Neutral_Bone",
]

ENROLLMENT_SIZES = [1, 2, 3, 5]

RANDOM_STATE = 42


# ============================================================
# LOAD FEATURES
# ============================================================

print("\n========================================")
print("MULTI-CONDITION ENROLLMENT EXPERIMENT")
print("========================================")

print("\nLoading feature table...")

df = pd.read_csv(FEATURE_FILE)

print("Rows:", len(df))
print("Columns:", len(df.columns))
print("Subjects:", df["subject"].nunique())
print("Conditions:", df["condition"].nunique())


# ============================================================
# IDENTIFY FEATURE COLUMNS
# ============================================================

# These are metadata/label columns and must NOT be used
# as classifier input.
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
    col for col in df.columns
    if col not in NON_FEATURE_COLUMNS
    and pd.api.types.is_numeric_dtype(df[col])
]

print("\nNumber of feature columns:", len(feature_columns))

if len(feature_columns) != 64:
    print(
        "WARNING: Expected 64 feature columns, "
        f"but found {len(feature_columns)}."
    )

print("\nFirst few features:")
print(feature_columns[:10])


# ============================================================
# VALIDATE DATASET
# ============================================================

print("\nCondition counts:")

condition_counts = df.groupby("condition").size()
print(condition_counts)

missing_conditions = [
    condition for condition in CONDITIONS
    if condition not in df["condition"].unique()
]

if missing_conditions:
    raise ValueError(
        f"Missing conditions in feature table: {missing_conditions}"
    )


# ============================================================
# MODEL
# ============================================================

def create_model():

    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


# ============================================================
# RUN ONE EXPERIMENT
# ============================================================

def evaluate_configuration(train_conditions, target_condition):

    train_df = df[
        df["condition"].isin(train_conditions)
    ].copy()

    test_df = df[
        df["condition"] == target_condition
    ].copy()

    X_train = train_df[feature_columns]
    y_train = train_df["subject"]

    X_test = test_df[feature_columns]
    y_test = test_df["subject"]

    model = create_model()

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    precision = precision_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0,
    )

    return {
        "enrollment_size": len(train_conditions),
        "train_conditions": "|".join(train_conditions),
        "target_condition": target_condition,
        "train_windows": len(train_df),
        "test_windows": len(test_df),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ============================================================
# MULTI-CONDITION EXPERIMENT
# ============================================================

results = []

for target_condition in CONDITIONS:

    print("\n========================================")
    print("UNSEEN TARGET:", target_condition)
    print("========================================")

    available_training_conditions = [
        condition
        for condition in CONDITIONS
        if condition != target_condition
    ]

    for enrollment_size in ENROLLMENT_SIZES:

        training_combinations = list(
            combinations(
                available_training_conditions,
                enrollment_size,
            )
        )

        print(
            f"\nEnrollment size {enrollment_size}: "
            f"{len(training_combinations)} combinations"
        )

        for i, train_conditions in enumerate(
            training_combinations,
            start=1,
        ):

            print(
                f"  [{i}/{len(training_combinations)}] "
                f"{' + '.join(train_conditions)} "
                f"-> {target_condition}"
            )

            result = evaluate_configuration(
                train_conditions,
                target_condition,
            )

            results.append(result)


# ============================================================
# SAVE DETAILED RESULTS
# ============================================================

results_df = pd.DataFrame(results)

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

results_df.to_csv(
    OUTPUT_FILE,
    index=False,
)

print("\n========================================")
print("DETAILED RESULTS SAVED")
print("========================================")

print(OUTPUT_FILE)


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
        mean_precision=("precision", "mean"),
        mean_recall=("recall", "mean"),
        mean_f1=("f1", "mean"),
    )
    .reset_index()
)

summary.to_csv(
    SUMMARY_FILE,
    index=False,
)


print("\n========================================")
print("OVERALL PERFORMANCE")
print("========================================")

display_summary = summary.copy()

for column in [
    "mean_accuracy",
    "std_accuracy",
    "mean_precision",
    "mean_recall",
    "mean_f1",
]:
    display_summary[column] *= 100

print(
    display_summary.round(2).to_string(
        index=False
    )
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

TARGET_SUMMARY_FILE = (
    ROOT
    / "results"
    / "tables"
    / "multicondition_target_summary.csv"
)

target_summary.to_csv(
    TARGET_SUMMARY_FILE,
    index=False,
)


print("\n========================================")
print("TARGET-WISE PERFORMANCE")
print("========================================")

target_display = target_summary.copy()

target_display["mean_accuracy"] *= 100
target_display["std_accuracy"] *= 100
target_display["mean_f1"] *= 100

print(
    target_display.round(2).to_string(
        index=False
    )
)


# ============================================================
# BEST CONFIGURATION FOR EACH ENROLLMENT SIZE
# ============================================================

print("\n========================================")
print("BEST CONFIGURATIONS")
print("========================================")

for size in ENROLLMENT_SIZES:

    subset = results_df[
        results_df["enrollment_size"] == size
    ]

    best = subset.loc[
        subset["accuracy"].idxmax()
    ]

    print(
        f"\nEnrollment size: {size}"
    )

    print(
        "Training:",
        best["train_conditions"],
    )

    print(
        "Target:",
        best["target_condition"],
    )

    print(
        f"Accuracy: {best['accuracy'] * 100:.2f}%"
    )

    print(
        f"Macro F1: {best['f1'] * 100:.2f}%"
    )


print("\n========================================")
print("EXPERIMENT COMPLETE")
print("========================================")

print("\nGenerated files:")

print(
    "1.",
    OUTPUT_FILE,
)

print(
    "2.",
    SUMMARY_FILE,
)

print(
    "3.",
    TARGET_SUMMARY_FILE,
)