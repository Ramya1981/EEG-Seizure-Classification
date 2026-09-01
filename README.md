# EEG Signal processing

A research-oriented Python project for preparing and analysing EEG data for epileptic seizure detection and classification. The repository currently focuses on EEG exploratory analysis, preprocessing, window generation, seizure/preictal labelling, multi-dataset loading, feature extraction, class balancing, and preparation of data for downstream machine-learning or deep-learning models.

## Project overview

EEG-based seizure analysis is challenging because recordings are high-dimensional, heterogeneous across datasets, and strongly class-imbalanced. This project provides a modular preprocessing pipeline for working with scalp EEG and intracranial EEG data and preparing balanced features for subsequent seizure-classification experiments.

The current code supports:

- Loading EDF recordings and seizure metadata from the CHB-MIT scalp EEG dataset
- Exploratory EEG analysis and power spectral density visualisation
- Band-pass filtering of EEG signals
- Resampling recordings to a common sampling frequency
- Segmenting continuous EEG into overlapping windows
- Generating seizure/preictal labels from available metadata
- Loading Mayo iEEG `.mat` files
- Combining data from different EEG sources
- Extracting simple statistical features
- Downsampling the majority class
- Applying SMOTE to reduce class imbalance

> **Development status:** This repository currently contains the data preparation and balancing pipeline. The final deep-learning classification model and evaluation workflow are still being developed and will be added in later versions.

## Datasets

### CHB-MIT Scalp EEG Database

The project contains code for working with the CHB-MIT scalp EEG database in EDF format. Subject summary files are parsed to obtain seizure information and recordings are processed into fixed-duration EEG windows.

Dataset files are **not included in this repository** because of their size. Users should obtain the dataset from its official source and configure the local dataset path before running the scripts.

### Mayo iEEG data

The repository also contains utilities for loading Mayo intracranial EEG recordings stored as MATLAB/HDF5 files. The loader resamples recordings from 400 Hz to 256 Hz before window generation.

These data files are also excluded from the repository.

## Processing pipeline

```text
Raw EEG recordings
        |
        v
Dataset loading and metadata parsing
        |
        v
EEG preprocessing / band-pass filtering
        |
        v
Resampling where required
        |
        v
Overlapping window generation
        |
        v
Seizure / preictal labelling
        |
        v
Feature extraction
        |
        v
Majority-class downsampling
        |
        v
Standardisation
        |
        v
SMOTE class balancing
        |
        v
Prepared dataset for classification
```

## Repository structure

```text
EEG-Seizure-Classification/
|
|-- main.py                  # Main data-processing and balancing workflow
|-- EDA.py                   # Exploratory analysis of CHB-MIT EDF recordings
|-- mayoEDA.py               # Exploratory analysis of Mayo iEEG recordings
|-- test.py                  # Utility script for inspecting MATLAB files
|-- requirements.txt         # Python package requirements
|
`-- modules/
    |-- __init__.py
    |-- eeg_loader.py        # EEG loading utility
    |-- eeg_preprocess.py    # EEG filtering/preprocessing
    |-- load_mayo.py         # Mayo iEEG loading and resampling
    |-- metadataloader.py    # CHB-MIT metadata/data loading
    |-- smote_utils.py       # SMOTE-based class balancing
    |-- summary_parser.py    # CHB-MIT summary-file parser
    `-- windowing.py         # EEG windowing and labelling utilities
```

## EEG preprocessing

The preprocessing module currently applies a fourth-order Butterworth band-pass filter between **0.5 and 70 Hz** with a default sampling frequency of **256 Hz**.

The main workflow uses overlapping EEG windows and subsequently derives simple statistical features (channel/window mean and variance) for class-balancing experiments.

## Class imbalance handling

Seizure EEG datasets commonly contain substantially more non-seizure than seizure-related observations. The current workflow addresses this by:

1. Downsampling the majority class relative to the minority-class size.
2. Standardising extracted features using `StandardScaler`.
3. Applying Synthetic Minority Over-sampling Technique (SMOTE).

## Installation

Clone the repository:

```bash
git clone https://github.com/Ramya1981/EEG-Seizure-Classification.git
cd EEG-Seizure-Classification
```

Create and activate a Python virtual environment, then install the dependencies:

```bash
pip install -r requirements.txt
```

## Running the project

Before running the scripts, update the local dataset paths in `main.py`, `EDA.py`, `mayoEDA.py`, `test.py`, and/or `modules/load_mayo.py` so they point to the datasets on your own computer.

The main workflow can then be started with:

```bash
python main.py
```

The exploratory CHB-MIT script can be run with:

```bash
python EDA.py
```

and Mayo exploratory analysis with:

```bash
python mayoEDA.py
```

## Important implementation note

The project is actively being refactored. Some dataset paths are currently configured as local Windows paths, and the window-labelling interfaces should be harmonised before treating the full `main.py` pipeline as a reproducible end-to-end release. Future updates will move dataset configuration outside the source files and add validation/tests for the complete pipeline.

## Technologies

- Python
- NumPy
- SciPy
- MNE-Python
- scikit-learn
- imbalanced-learn
- h5py
- Matplotlib

## Future work

Planned improvements include:

- Removing hard-coded dataset paths and adding configuration options
- Harmonising CHB-MIT and Mayo data representations
- Improving seizure/preictal labelling and validation
- Adding automated tests
- Adding deep-learning classification models
- Reporting evaluation metrics such as accuracy, sensitivity, specificity, precision, recall, F1-score and ROC-AUC
- Adding reproducible experiment configurations and result visualisations

## Data and repository policy

Large EEG recordings and generated binary data are intentionally excluded from version control. The `.gitignore` prevents common EEG/raw-data formats and local generated files from being committed to the public repository.

## Research use

This repository is intended for research and educational use in EEG signal analysis and epileptic seizure modelling. It is not intended for clinical diagnosis or medical decision-making.
