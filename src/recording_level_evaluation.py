import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from config import TABLES_DIR, RANDOM_STATE


# ---------------------------------------------------------
# Settings
# ---------------------------------------------------------

FEATURE_FILE = TABLES_DIR / "eeg_features.csv"

CONDITIONS = [
    "Native_InEar",
    "NonNative_InEar",
    "Neutral_InEar",
    "Native_Bone",
    "NonNative_Bone",
    "Neutral_Bone",
]

TRAIN_WINDOWS = 17

# Number of consecutive 4-second windows used
# for one identification decision
VOTING_WINDOWS = [1, 3, 5, 10]


# ---------------------------------------------------------
# Load features
# ---------------------------------------------------------

print("Loading feature table...")

df = pd.read_csv(FEATURE_FILE)

metadata_columns = [
    "subject",
    "experiment",
    "condition",
    "recording",
    "window_number",
    "window_start_sample",
]

feature_columns = [
    col for col in df.columns
    if col not in metadata_columns
]

print(f"Rows: {len(df)}")
print(f"Subjects: {df['subject'].nunique()}")
print(f"Features: {len(feature_columns)}")


# ---------------------------------------------------------
# Majority vote
# ---------------------------------------------------------

def majority_vote(predictions):
    """
    Return the most frequently predicted subject.
    """

    values, counts = np.unique(
        predictions,
        return_counts=True
    )

    return values[np.argmax(counts)]


# ---------------------------------------------------------
# Evaluate predictions in consecutive groups
# ---------------------------------------------------------

def evaluate_segments(test_df, predictions, group_size):

    temp = test_df[
        [
            "subject",
            "recording",
            "window_number"
        ]
    ].copy()

    temp["prediction"] = predictions

    temp = temp.sort_values(
        ["recording", "window_number"]
    )

    correct = 0
    total = 0

    # Evaluate each recording independently
    for recording, recording_df in temp.groupby("recording"):

        recording_df = recording_df.sort_values(
            "window_number"
        )

        true_subject = recording_df["subject"].iloc[0]

        preds = recording_df["prediction"].to_numpy()

        # Divide recording into NON-OVERLAPPING groups
        for start in range(0, len(preds), group_size):

            segment = preds[start:start + group_size]

            # Ignore incomplete final segment
            if len(segment) < group_size:
                continue

            voted_subject = majority_vote(segment)

            if voted_subject == true_subject:
                correct += 1

            total += 1

    if total == 0:
        return np.nan, 0

    return correct / total, total


# ---------------------------------------------------------
# Run cross-condition evaluation
# ---------------------------------------------------------

results = []

for source_condition in CONDITIONS:

    source_df = df[
        df["condition"] == source_condition
    ].copy()

    # First 17 windows from every source recording
    train_df = (
        source_df
        .sort_values(["recording", "window_number"])
        .groupby("recording", group_keys=False)
        .head(TRAIN_WINDOWS)
    )

    X_train = train_df[feature_columns]
    y_train = train_df["subject"]

    print("\n======================================")
    print(f"Source: {source_condition}")
    print(f"Training windows: {len(train_df)}")
    print("======================================")

    for target_condition in CONDITIONS:

        # Cross-condition only.
        # Avoid source == target because the first 17
        # windows would overlap with training data.
        if source_condition == target_condition:
            continue

        test_df = df[
            df["condition"] == target_condition
        ].copy()

        test_df = test_df.sort_values(
            ["recording", "window_number"]
        )

        X_test = test_df[feature_columns]

        # ---------------------------------------------
        # Train Random Forest
        # ---------------------------------------------

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(
                n_estimators=200,
                random_state=RANDOM_STATE,
                n_jobs=-1
            ))
        ])

        model.fit(
            X_train,
            y_train
        )

        predictions = model.predict(
            X_test
        )

        # ---------------------------------------------
        # Evaluate different amounts of EEG
        # ---------------------------------------------

        for group_size in VOTING_WINDOWS:

            accuracy, decisions = evaluate_segments(
                test_df,
                predictions,
                group_size
            )

            duration_seconds = (
                group_size * 4
            )

            results.append({
                "source_condition": source_condition,
                "target_condition": target_condition,
                "model": "Random Forest",
                "windows_per_decision": group_size,
                "duration_seconds": duration_seconds,
                "accuracy": accuracy,
                "number_of_decisions": decisions,
            })

            print(
                f"{target_condition:20s} | "
                f"{group_size:2d} windows "
                f"({duration_seconds:2d}s) | "
                f"Accuracy: {accuracy * 100:6.2f}% | "
                f"Decisions: {decisions}"
            )


# ---------------------------------------------------------
# Save detailed results
# ---------------------------------------------------------

results_df = pd.DataFrame(results)

output_file = (
    TABLES_DIR /
    "recording_level_results.csv"
)

results_df.to_csv(
    output_file,
    index=False
)


# ---------------------------------------------------------
# Overall summary
# ---------------------------------------------------------

summary = (
    results_df
    .groupby(
        [
            "windows_per_decision",
            "duration_seconds"
        ]
    )
    .agg(
        mean_accuracy=("accuracy", "mean"),
        std_accuracy=("accuracy", "std"),
        total_decisions=("number_of_decisions", "sum")
    )
    .reset_index()
)

summary["mean_accuracy"] *= 100
summary["std_accuracy"] *= 100


print("\n======================================")
print("OVERALL RANDOM FOREST RESULTS")
print("======================================")

print(
    summary.round(2).to_string(index=False)
)


# ---------------------------------------------------------
# Save summary
# ---------------------------------------------------------

summary_file = (
    TABLES_DIR /
    "recording_level_summary.csv"
)

summary.to_csv(
    summary_file,
    index=False
)

print("\nDetailed results saved to:")
print(output_file)

print("\nSummary saved to:")
print(summary_file)