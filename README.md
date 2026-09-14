# Robust EEG-Based Biometric Authentication Across Auditory Stimulation Conditions

This repository contains the code and experimental results for a study investigating the robustness of EEG-based biometric identification across different auditory stimulation conditions.

The experiments use the publicly available **Auditory Evoked Potential EEG-Biometric Dataset** from PhysioNet.

## Research Question

Can subject-specific EEG characteristics remain sufficiently discriminative when the auditory condition used during testing differs from the condition used during training?

## Dataset

The study uses EEG recordings from:

- 20 subjects
- 4 EEG channels: P4, Cz, F8, T7
- Sampling frequency: 200 Hz
- 6 auditory conditions

The auditory conditions are:

| Experiment | Condition |
|---|---|
| ex05 | Native In-Ear |
| ex06 | Non-Native In-Ear |
| ex07 | Neutral In-Ear |
| ex08 | Native Bone |
| ex09 | Non-Native Bone |
| ex10 | Neutral Bone |

The dataset is **not included in this repository**.

Download the Auditory Evoked Potential EEG-Biometric Dataset from PhysioNet and place it inside the `data/` directory.

Expected directory structure:

```text
data/
└── physionet.org/
    └── files/
        └── auditory-eeg/
            └── 1.0.0/
                └── Filtered_Data/

```
## Preprocessing and Segmentation

The experiments use the filtered EEG recordings provided with the dataset.

Each EEG recording is divided into non-overlapping 4-second windows. At a sampling frequency of 200 Hz, each window contains 800 samples per channel.

To ensure equal representation across recordings, 29 complete windows are retained from each recording, corresponding to the shortest recording available in the selected conditions.

This results in:

20 subjects × 6 conditions × 29 windows = 3,480 EEG windows.

## Feature Extraction

A total of 64 features are extracted from each EEG window.

For each of the four EEG channels, 16 features are calculated.

### Time-Domain Features

- Mean
- Standard deviation
- Variance
- Root Mean Square (RMS)
- Skewness
- Kurtosis
- Minimum
- Maximum

### Frequency-Domain Features

Absolute and relative spectral power are calculated for four EEG frequency bands:

- Delta: 1–4 Hz
- Theta: 4–8 Hz
- Alpha: 8–13 Hz
- Beta: 13–30 Hz

Power Spectral Density (PSD) is estimated using Welch's method.

## Machine Learning Models

Three machine-learning classifiers are evaluated:

- Support Vector Machine (SVM) with RBF kernel
- Random Forest (RF)
- K-Nearest Neighbors (KNN)

The experimental task is formulated as closed-set 20-class subject identification.

## Experimental Design

### Same-Condition Identification

The model is trained and tested using EEG recorded under the same auditory condition.

For each recording:

- First 17 windows are used for training.
- Remaining 12 windows are used for testing.

### Cross-Condition Identification

The model is trained using EEG from one auditory condition and evaluated using EEG from a different auditory condition.

All directed pairs among the six auditory conditions are evaluated.

This produces 30 cross-condition combinations for each classifier.

### Transducer Analysis

Cross-condition performance is additionally compared between:

- Same-transducer transfer
- Cross-transducer transfer

This evaluates the effect of changing between In-Ear and Bone conduction.

### Temporal Aggregation

Random Forest predictions from consecutive EEG windows are combined using majority voting.

The following decision durations are evaluated:

| Windows per Decision | EEG Duration |
|---:|---:|
| 1 | 4 s |
| 3 | 12 s |
| 5 | 20 s |
| 10 | 40 s |

## Results

### Average Same-Condition Performance

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| Random Forest | 77.99% | 77.36% |
| SVM | 76.67% | 76.30% |
| KNN | 69.10% | 67.72% |

### Average Cross-Condition Performance

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| Random Forest | 68.24% | 66.78% |
| SVM | 65.59% | 64.66% |
| KNN | 60.75% | 59.43% |

Random Forest achieved the strongest overall performance.

Its mean accuracy decreased from 77.99% under same-condition evaluation to 68.24% under cross-condition evaluation, indicating that auditory-condition changes affect EEG-based subject identification.

### Effect of Temporal Aggregation

| EEG Duration | Mean RF Cross-Condition Accuracy |
|---:|---:|
| 4 s | 68.24% |
| 12 s | 76.04% |
| 20 s | 82.00% |
| 40 s | 87.08% |

Temporal aggregation substantially improved cross-condition identification performance.

## Repository Structure

```text
.
├── src/
│   ├── config.py
│   ├── feature_extraction.py
│   ├── train_models.py
│   ├── cross_condition.py
│   ├── analyze_results.py
│   ├── create_heatmaps.py
│   ├── recording_level_evaluation.py
│   └── plot_duration_accuracy.py
│
├── results/
│   ├── figures/
│   └── tables/
│
├── .gitignore
├── README.md
└── requirements.txt

```
## Reproducing the Experiments

### 1. Clone the Repository

```bash
git clone https://github.com/aspaankaur/eeg-biometric-authentication.git
cd eeg-biometric-authentication
```

### 2. Install Dependencies

It is recommended to use a virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For Windows PowerShell:

```bash
.venv\Scripts\Activate.ps1
```

### 3. Download the Dataset

Download the **Auditory Evoked Potential EEG-Biometric Dataset** from PhysioNet.

The dataset itself is not included in this repository.

Place the downloaded dataset so that the filtered recordings are available at:

```text
data/
└── physionet.org/
    └── files/
        └── auditory-eeg/
            └── 1.0.0/
                └── Filtered_Data/
```

### 4. Run the Experiment Pipeline

Run the scripts in the following order:

```bash
python src/feature_extraction.py
python src/train_models.py
python src/cross_condition.py
python src/analyze_results.py
python src/create_heatmaps.py
python src/recording_level_evaluation.py
python src/plot_duration_accuracy.py
```

The generated result tables are stored in:

```text
results/tables/
```

and the generated figures are stored in:

```text
results/figures/
```

## Important Methodological Note

This work investigates EEG biometrics in the broader context of biometric authentication. However, the experiments reported in this repository perform **closed-set 20-class subject identification**.

The current experiments do not perform claimed-identity 1:1 biometric verification. Therefore, biometric verification metrics such as False Acceptance Rate (FAR), False Rejection Rate (FRR), and Equal Error Rate (EER) are not reported.

## Limitations

The study uses a relatively small dataset of 20 subjects and four EEG channels. The recordings were collected under controlled experimental conditions.

The results should therefore be interpreted as an experimental evaluation of cross-condition EEG biometric robustness rather than as evidence of deployment-ready biometric authentication performance.

## Citation

This project uses the **Auditory Evoked Potential EEG-Biometric Dataset** available through PhysioNet.

If you use the dataset in academic work, please cite the original dataset and its associated publication.

## Author

**Aspaan Kaur**  
Department of Computer Science and Engineering  
Gopalan College of Engineering and Management  
Bengaluru, India