from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import skew, kurtosis

from src.config import (
    FILTERED_DIR,
    TABLES_DIR,
    FS,
    CHANNELS,
    WINDOW_SAMPLES,
    CONDITIONS,
)


# --------------------------------------------------
# Feature functions
# --------------------------------------------------

def time_domain_features(signal):
    """Extract basic statistical features from one EEG window."""

    return {
        "mean": np.mean(signal),
        "std": np.std(signal),
        "variance": np.var(signal),
        "rms": np.sqrt(np.mean(signal ** 2)),
        "skewness": skew(signal),
        "kurtosis": kurtosis(signal),
        "min": np.min(signal),
        "max": np.max(signal),
    }


def bandpower(freqs, psd, low, high):
    """Calculate absolute power in a frequency band."""

    mask = (freqs >= low) & (freqs < high)

    if not np.any(mask):
        return 0.0

    return np.trapezoid(psd[mask], freqs[mask])


def frequency_features(signal):
    """Extract EEG frequency-band features."""

    freqs, psd = welch(
        signal,
        fs=FS,
        nperseg=min(256, len(signal)),
    )

    bands = {
        "delta": (1, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
    }

    powers = {}

    for band_name, (low, high) in bands.items():
        powers[band_name] = bandpower(
            freqs,
            psd,
            low,
            high,
        )

    total_power = sum(powers.values())

    features = {}

    for band_name, power in powers.items():
        features[f"{band_name}_power"] = power

        if total_power > 0:
            features[f"{band_name}_relative"] = (
                power / total_power
            )
        else:
            features[f"{band_name}_relative"] = 0.0

    return features


# --------------------------------------------------
# Filename parsing
# --------------------------------------------------

def parse_filename(filename):
    """
    Parse filenames such as:
    s14_ex09.csv
    """

    pattern = re.compile(
        r"s(\d+)_ex(\d+)(?:_s(\d+))?\.csv$"
    )

    match = pattern.match(filename)

    if not match:
        return None

    subject = int(match.group(1))
    experiment = int(match.group(2))

    if experiment not in CONDITIONS:
        return None

    return subject, experiment


# --------------------------------------------------
# Extract features from one recording
# --------------------------------------------------

def extract_recording_features(filepath):
    parsed = parse_filename(filepath.name)

    if parsed is None:
        return []

    subject, experiment = parsed
    condition = CONDITIONS[experiment]

    print(f"Processing {filepath.name} → {condition}")

    # First column is the sample index.
    data = pd.read_csv(filepath, index_col=0)

    # Keep only the four EEG channels.
    data = data[CHANNELS]

    # Use the same number of complete windows from every recording.
    # The shortest recording contains 29 complete 4-second windows.
    # Therefore, 29 windows (23,200 samples) are retained per recording.
    MAX_WINDOWS = 29
    data = data.iloc[:MAX_WINDOWS * WINDOW_SAMPLES]
    
    signal_length = len(data)

    rows = []

    window_number = 0

    for start in range(
        0,
        signal_length - WINDOW_SAMPLES + 1,
        WINDOW_SAMPLES,
    ):
        end = start + WINDOW_SAMPLES

        window = data.iloc[start:end]

        features = {
            "subject": subject,
            "experiment": experiment,
            "condition": condition,
            "recording": filepath.stem,
            "window_number": window_number,
            "window_start_sample": start,
        }

        # Extract features from every EEG channel.
        for channel in CHANNELS:

            signal = window[channel].to_numpy(
                dtype=float
            )

            time_features = time_domain_features(signal)
            freq_features = frequency_features(signal)

            for name, value in time_features.items():
                features[f"{channel}_{name}"] = value

            for name, value in freq_features.items():
                features[f"{channel}_{name}"] = value

        rows.append(features)

        window_number += 1

    return rows


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    print("EEG Feature Extraction")
    print("======================")
    print(f"Dataset: {FILTERED_DIR}")
    print()

    TABLES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_rows = []

    files = sorted(
        FILTERED_DIR.glob("*.csv")
    )

    # Only process ex05–ex10.
    selected_files = []

    for filepath in files:

        parsed = parse_filename(filepath.name)

        if parsed is not None:
            selected_files.append(filepath)

    print(
        f"Found {len(selected_files)} recordings "
        "for ex05–ex10."
    )
    print()

    for filepath in selected_files:

        rows = extract_recording_features(filepath)

        all_rows.extend(rows)

    features_df = pd.DataFrame(all_rows)

    output_file = (
        TABLES_DIR / "eeg_features.csv"
    )

    features_df.to_csv(
        output_file,
        index=False,
    )

    print()
    print("======================")
    print("Feature extraction complete!")
    print(
        f"Total feature rows: {len(features_df)}"
    )
    print(
        f"Total columns: {len(features_df.columns)}"
    )
    print(
        f"Saved to: {output_file}"
    )


if __name__ == "__main__":
    main()