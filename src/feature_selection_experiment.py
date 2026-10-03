"""
Condition-robust feature-selection experiment.

For each unseen target condition:
1. Train using the other five auditory conditions.
2. Rank features using Random Forest feature importance on TRAINING DATA ONLY.
3. Evaluate Top 10, Top 20, Top 30, and all 64 features.
4. Test exclusively on the unseen target condition.
"""

from pathlib import Path
from collections import Counter

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

FEATURE_FILE = (
    ROOT
    / "results"
    / "tables"
    / "eeg_features.csv"
)

RESULT_FILE = (
    ROOT
    / "results"
    / "tables"
    / "feature_selection_results.csv"
)

RANKING_FILE = (
    ROOT
    / "results"
    / "tables"
    / "feature_rankings.csv"
)

STABILITY_FILE = (
    ROOT
    / "results"
    / "tables"
    / "feature_stability.csv"
)


# ============================================================
# SETTINGS
# ============================================================

CONDITIONS = [
    "Native_InEar",
    "NonNative_InEar",
    "Neutral_InEar",
    "Native_Bone",
    "NonNative_Bone",
    "Neutral_Bone",
]

FEATURE_SET_SIZES = [10, 20, 30, 64]

RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

print("\n========================================")
print("CONDITION-ROBUST FEATURE SELECTION")
print("========================================")

df = pd.read_csv(FEATURE_FILE)

print("\nRows:", len(df))
print("Columns:", len(df.columns))
print("Subjects:", df["subject"].nunique())
print("Conditions:", df["condition"].nunique())


# ============================================================
# FIND FEATURE COLUMNS
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

print("\nFeature columns:", len(feature_columns))

if len(feature_columns) != 64:
    raise ValueError(
        "Expected 64 feature columns, "
        f"but found {len(feature_columns)}."
    )


# ============================================================
# MODEL
# ============================================================

def make_rf():

    return RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


# ============================================================
# STORAGE
# ============================================================

results = []
rankings = []

top10_counter = Counter()
top20_counter = Counter()
top30_counter = Counter()


# ============================================================
# LEAVE-ONE-CONDITION-OUT EXPERIMENT
# ============================================================

for target_condition in CONDITIONS:

    print("\n========================================")
    print("UNSEEN TARGET:", target_condition)
    print("========================================")

    train_df = df[
        df["condition"] != target_condition
    ].copy()

    test_df = df[
        df["condition"] == target_condition
    ].copy()

    X_train_all = train_df[feature_columns]
    y_train = train_df["subject"]

    X_test_all = test_df[feature_columns]
    y_test = test_df["subject"]

    print(
        "Training windows:",
        len(train_df)
    )

    print(
        "Testing windows:",
        len(test_df)
    )


    # ========================================================
    # FEATURE RANKING
    # ========================================================

    ranking_model = make_rf()

    ranking_model.fit(
        X_train_all,
        y_train,
    )

    importance_df = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance":
                ranking_model.feature_importances_,
        }
    )

    importance_df = (
        importance_df
        .sort_values(
            "importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    importance_df["rank"] = (
        importance_df.index + 1
    )

    importance_df[
        "target_condition"
    ] = target_condition

    rankings.append(
        importance_df
    )


    # ========================================================
    # STABILITY COUNTS
    # ========================================================

    top10 = (
        importance_df
        .head(10)["feature"]
        .tolist()
    )

    top20 = (
        importance_df
        .head(20)["feature"]
        .tolist()
    )

    top30 = (
        importance_df
        .head(30)["feature"]
        .tolist()
    )

    top10_counter.update(top10)
    top20_counter.update(top20)
    top30_counter.update(top30)


    # ========================================================
    # EVALUATE FEATURE SUBSETS
    # ========================================================

    for feature_set_size in FEATURE_SET_SIZES:

        if feature_set_size == 64:

            selected_features = (
                feature_columns
            )

        else:

            selected_features = (
                importance_df
                .head(feature_set_size)
                ["feature"]
                .tolist()
            )

        print(
            f"Testing Top "
            f"{feature_set_size} features..."
        )

        X_train = (
            train_df[
                selected_features
            ]
        )

        X_test = (
            test_df[
                selected_features
            ]
        )

        model = Pipeline(
            [
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "classifier",
                    make_rf(),
                ),
            ]
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

        results.append(
            {
                "target_condition":
                    target_condition,

                "feature_set_size":
                    feature_set_size,

                "accuracy":
                    accuracy,

                "precision":
                    precision,

                "recall":
                    recall,

                "f1":
                    f1,
            }
        )


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df.to_csv(
    RESULT_FILE,
    index=False,
)


# ============================================================
# SAVE FEATURE RANKINGS
# ============================================================

rankings_df = pd.concat(
    rankings,
    ignore_index=True,
)

rankings_df.to_csv(
    RANKING_FILE,
    index=False,
)


# ============================================================
# FEATURE STABILITY
# ============================================================

stability_rows = []

for feature in feature_columns:

    stability_rows.append(
        {
            "feature":
                feature,

            "top10_count":
                top10_counter[feature],

            "top20_count":
                top20_counter[feature],

            "top30_count":
                top30_counter[feature],
        }
    )

stability_df = pd.DataFrame(
    stability_rows
)

stability_df = (
    stability_df
    .sort_values(
        [
            "top10_count",
            "top20_count",
            "top30_count",
        ],
        ascending=False,
    )
)

stability_df.to_csv(
    STABILITY_FILE,
    index=False,
)


# ============================================================
# SUMMARY
# ============================================================

summary = (
    results_df
    .groupby(
        "feature_set_size"
    )
    .agg(
        mean_accuracy=(
            "accuracy",
            "mean",
        ),

        std_accuracy=(
            "accuracy",
            "std",
        ),

        mean_precision=(
            "precision",
            "mean",
        ),

        mean_recall=(
            "recall",
            "mean",
        ),

        mean_f1=(
            "f1",
            "mean",
        ),
    )
    .reset_index()
)


print("\n========================================")
print("FEATURE-SELECTION PERFORMANCE")
print("========================================")

display_summary = (
    summary.copy()
)

for column in [
    "mean_accuracy",
    "std_accuracy",
    "mean_precision",
    "mean_recall",
    "mean_f1",
]:

    display_summary[column] *= 100


print(
    display_summary
    .round(2)
    .to_string(index=False)
)


# ============================================================
# TARGET-WISE RESULTS
# ============================================================

print("\n========================================")
print("TARGET-WISE ACCURACY")
print("========================================")

pivot = results_df.pivot(
    index="target_condition",
    columns="feature_set_size",
    values="accuracy",
)

pivot *= 100

print(
    pivot
    .round(2)
    .to_string()
)


# ============================================================
# MOST STABLE FEATURES
# ============================================================

print("\n========================================")
print("MOST STABLE TOP FEATURES")
print("========================================")

print(
    stability_df
    .head(20)
    .to_string(index=False)
)


print("\n========================================")
print("EXPERIMENT COMPLETE")
print("========================================")

print("\nSaved:")

print(
    RESULT_FILE
)

print(
    RANKING_FILE
)

print(
    STABILITY_FILE
)