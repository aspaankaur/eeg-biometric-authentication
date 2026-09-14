from pathlib import Path

import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from src.config import TABLES_DIR, FIGURES_DIR, RANDOM_STATE


# --------------------------------------------------
# Configuration
# --------------------------------------------------

FEATURE_FILE = TABLES_DIR / "eeg_features.csv"

# Number of windows used for training.
# Each recording has 29 windows.
TRAIN_WINDOWS = 17

MODELS = {
    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(
            kernel="rbf",
            C=10,
            random_state=RANDOM_STATE
        )),
    ]),

    "Random Forest": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )),
    ]),

    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(
            n_neighbors=5
        )),
    ]),
}


# --------------------------------------------------
# Load data
# --------------------------------------------------

def load_data():
    print("Loading feature dataset...")

    df = pd.read_csv(FEATURE_FILE)

    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"Subjects: {df['subject'].nunique()}")
    print(f"Recordings: {df['recording'].nunique()}")

    return df


# --------------------------------------------------
# Identify feature columns
# --------------------------------------------------

def get_feature_columns(df):
    metadata_columns = [
        "subject",
        "experiment",
        "condition",
        "recording",
        "window_number",
        "window_start_sample",
    ]

    return [
        column
        for column in df.columns
        if column not in metadata_columns
    ]


# --------------------------------------------------
# Create same-condition train/test split
# --------------------------------------------------

def split_same_condition(df):
    """
    For every recording:

    First 17 windows  -> training
    Remaining 12      -> testing

    This is a chronological split, not a random window split.
    """

    train_parts = []
    test_parts = []

    for recording, recording_df in df.groupby("recording"):

        recording_df = recording_df.sort_values(
            "window_number"
        )

        train_parts.append(
            recording_df.iloc[:TRAIN_WINDOWS]
        )

        test_parts.append(
            recording_df.iloc[TRAIN_WINDOWS:]
        )

    train_df = pd.concat(train_parts)
    test_df = pd.concat(test_parts)

    return train_df, test_df


# --------------------------------------------------
# Evaluate one model
# --------------------------------------------------

def evaluate_model(model, X_train, y_train, X_test, y_test):

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    return {
        "accuracy": accuracy_score(
            y_test,
            predictions
        ),
        "precision": precision_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0
        ),
        "recall": recall_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0
        ),
        "f1": f1_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0
        ),
    }


# --------------------------------------------------
# Same-condition experiment
# --------------------------------------------------

def run_same_condition_experiment(df):

    feature_columns = get_feature_columns(df)

    results = []

    conditions = sorted(
        df["condition"].unique()
    )

    for condition in conditions:

        print()
        print("=" * 60)
        print(f"Condition: {condition}")
        print("=" * 60)

        condition_df = df[
            df["condition"] == condition
        ].copy()

        train_df, test_df = split_same_condition(
            condition_df
        )

        X_train = train_df[feature_columns]
        y_train = train_df["subject"]

        X_test = test_df[feature_columns]
        y_test = test_df["subject"]

        print(
            f"Training windows: {len(train_df)}"
        )
        print(
            f"Testing windows: {len(test_df)}"
        )

        for model_name, model in MODELS.items():

            print(f"Running {model_name}...")

            metrics = evaluate_model(
                model,
                X_train,
                y_train,
                X_test,
                y_test
            )

            results.append({
                "experiment": "same_condition",
                "condition": condition,
                "model": model_name,
                **metrics,
            })

            print(
                f"  Accuracy: {metrics['accuracy']:.4f}"
            )
            print(
                f"  Macro F1:  {metrics['f1']:.4f}"
            )

    return pd.DataFrame(results)


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    print()
    print("EEG BIOMETRIC AUTHENTICATION")
    print("Same-Condition Baseline")
    print("=" * 60)

    df = load_data()

    results = run_same_condition_experiment(df)

    output_file = (
        TABLES_DIR /
        "same_condition_results.csv"
    )

    results.to_csv(
        output_file,
        index=False
    )

    print()
    print("=" * 60)
    print("Experiment complete!")
    print("=" * 60)

    print(results.to_string(index=False))

    print()
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
