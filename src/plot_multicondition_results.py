from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]

SUMMARY_FILE = (
    ROOT
    / "results"
    / "tables"
    / "multicondition_enrollment_summary.csv"
)

OUTPUT_FILE = (
    ROOT
    / "results"
    / "figures"
    / "accuracy_vs_enrollment_conditions.png"
)


df = pd.read_csv(SUMMARY_FILE)

x = df["enrollment_size"]
accuracy = df["mean_accuracy"] * 100


plt.figure(figsize=(7, 5))

plt.plot(
    x,
    accuracy,
    marker="o",
    linewidth=2
)

for x_value, y_value in zip(x, accuracy):
    plt.text(
        x_value,
        y_value + 0.5,
        f"{y_value:.2f}%",
        ha="center"
    )

plt.xlabel("Number of Enrollment Conditions")
plt.ylabel("Mean Identification Accuracy (%)")

plt.title(
    "Effect of Multi-Condition Enrollment on\n"
    "Unseen-Condition EEG Identification"
)

plt.xticks([1, 2, 3, 5])

plt.ylim(
    min(accuracy) - 3,
    max(accuracy) + 4
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

plt.savefig(
    OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved:", OUTPUT_FILE)