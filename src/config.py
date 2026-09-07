from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "auditory-eeg" / "1.0.0"
FILTERED_DIR = DATA_DIR / "Filtered_Data"

RESULTS_DIR = PROJECT_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"

# EEG settings
FS = 200
CHANNELS = ["T7", "F8", "Cz", "P4"]

# Windowing
WINDOW_SECONDS = 4
WINDOW_SAMPLES = FS * WINDOW_SECONDS

# Auditory conditions
CONDITIONS = {
    5: "Native_InEar",
    6: "NonNative_InEar",
    7: "Neutral_InEar",
    8: "Native_Bone",
    9: "NonNative_Bone",
    10: "Neutral_Bone",
}

# Models
RANDOM_STATE = 42
