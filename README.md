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