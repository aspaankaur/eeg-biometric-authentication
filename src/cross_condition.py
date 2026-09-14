import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

print("Loading feature table...")

df = pd.read_csv(FEATURE_FILE)

print(f"Total rows: {len(df)}")
print(f"Total recordings: {df['recording'].nunique()}")
print(f"Total subjects: {df['subject'].nunique()}")


# ---------------------------------------------------------
# Identify feature columns
# ---------------------------------------------------------

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

print(f"Number of features: {len(feature_columns)}")


# ---------------------------------------------------------
# Prepare models
# ---------------------------------------------------------

models = {
    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(
            C=10,
            kernel="rbf"
        ))
    ]),

    "Random Forest": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ]),

    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(
            n_neighbors=5
        ))
    ]),
}


# ---------------------------------------------------------
# Get training windows from one condition
# ---------------------------------------------------------

def get_training_data(condition):
    """
    Use the first 17 windows from every recording
    in the source condition.
    """

    condition_df = df[df["condition"] == condition].copy()

    train_df = (
        condition_df
        .sort_values(["recording", "window_number"])
        .groupby("recording", group_keys=False)
        .head(TRAIN_WINDOWS)
    )

    X = train_df[feature_columns]
    y = train_df["subject"]

    return X, y


# ---------------------------------------------------------
# Get all test windows from another condition
# ---------------------------------------------------------

def get_test_data(condition):
    """
    Use all 29 windows from every recording
    in the target condition.
    """

    condition_df = df[df["condition"] == condition].copy()

    condition_df = condition_df.sort_values(
        ["recording", "window_number"]
    )

    X = condition_df[feature_columns]
    y = condition_df["subject"]

    return X, y


# ---------------------------------------------------------
# Run cross-condition experiment
# ---------------------------------------------------------

results = []

for source_condition in CONDITIONS:

    print(f"\nSource condition: {source_condition}")

    X_train, y_train = get_training_data(source_condition)

    print(f"Training windows: {len(X_train)}")

    for target_condition in CONDITIONS:

        # -------------------------------------------------
        # Do NOT run source == target here.
        #
        # If source and target are identical, the training
        # windows would overlap with the test windows.
        #
        # Same-condition results already exist in:
        # same_condition_results.csv
        # -------------------------------------------------

        if source_condition == target_condition:
            continue

        X_test, y_test = get_test_data(target_condition)

        print(
            f"  Testing on {target_condition}: "
            f"{len(X_test)} windows"
        )

        for model_name, model in models.items():

            model.fit(X_train, y_train)

            predictions = model.predict(X_test)

            accuracy = accuracy_score(
                y_test,
                predictions
            )

            precision = precision_score(
                y_test,
                predictions,
                average="macro",
                zero_division=0
            )

            recall = recall_score(
                y_test,
                predictions,
                average="macro",
                zero_division=0
            )

            f1 = f1_score(
                y_test,
                predictions,
                average="macro",
                zero_division=0
            )

            results.append({
                "experiment": "cross_condition",
                "source_condition": source_condition,
                "target_condition": target_condition,
                "model": model_name,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            })


# ---------------------------------------------------------
# Save results
# ---------------------------------------------------------

results_df = pd.DataFrame(results)

output_file = TABLES_DIR / "cross_condition_results.csv"

results_df.to_csv(
    output_file,
    index=False
)

print("\n-----------------------------------------")
print("Cross-condition experiment complete!")
print("-----------------------------------------")

print(f"Results saved to: {output_file}")

print(f"Total experiments: {len(results_df)}")

print("\nAverage accuracy by model:")

print(
    results_df
    .groupby("model")["accuracy"]
    .mean()
    .sort_values(ascending=False)
)

print("\nAverage F1-score by model:")

print(
    results_df
    .groupby("model")["f1"]
    .mean()
    .sort_values(ascending=False)
)