import pandas as pd
import matplotlib.pyplot as plt

from config import TABLES_DIR, FIGURES_DIR


# ---------------------------------------------------------
# Load summary
# ---------------------------------------------------------

summary_file = TABLES_DIR / "recording_level_summary.csv"
df = pd.read_csv(summary_file)

FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Plot
# ---------------------------------------------------------

plt.figure(figsize=(7, 5))

plt.plot(
    df["duration_seconds"],
    df["mean_accuracy"],
    marker="o",
    linewidth=2
)

# Add accuracy labels above each point
for _, row in df.iterrows():
    plt.annotate(
        f"{row['mean_accuracy']:.2f}%",
        (
            row["duration_seconds"],
            row["mean_accuracy"]
        ),
        textcoords="offset points",
        xytext=(0, 8),
        ha="center"
    )


plt.xlabel("EEG Duration per Identification Decision (seconds)")
plt.ylabel("Mean Cross-Condition Accuracy (%)")

plt.title(
    "Effect of Temporal Aggregation on\n"
    "Cross-Condition EEG Identification"
)

plt.xticks(
    df["duration_seconds"]
)

plt.ylim(
    max(0, df["mean_accuracy"].min() - 10),
    min(100, df["mean_accuracy"].max() + 8)
)

plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.4
)

plt.tight_layout()


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

output_file = (
    FIGURES_DIR /
    "accuracy_vs_eeg_duration.png"
)

plt.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Figure generated successfully.")
print(f"Saved to: {output_file}")

print("\nValues used:")

print(
    df[
        [
            "duration_seconds",
            "mean_accuracy",
            "std_accuracy",
            "total_decisions"
        ]
    ].to_string(index=False)
)
