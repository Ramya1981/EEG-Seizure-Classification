import os

import mne
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample as sklearn_resample

from modules.eeg_preprocess import preprocess_signal
from modules.load_mayo import load_mayo_dataset
from modules.metadataloader import load_chbmit_seizure_metadata
from modules.smote_utils import apply_smote
from modules.windowing import (
    label_windows_with_preictal,
    window_eeg_signal,
)


# ---------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------
def extract_features(windows):
    """
    Extract simple statistical features from each EEG window.

    For every channel, the mean and variance are calculated and
    concatenated into a single feature vector.
    """
    features = []

    for window in windows:
        mean = np.mean(window, axis=1)
        var = np.var(window, axis=1)

        feat = np.concatenate([mean, var])
        features.append(feat)

    return np.array(features)


# ---------------------------------------------------------------------
# CHB-MIT data loading
# ---------------------------------------------------------------------
def load_chbmit_dataset(
    base_dir,
    metadata,
    window_sec=10,
    preictal_sec=300
):
    """
    Load CHB-MIT EEG recordings, preprocess them, generate windows,
    and assign binary preictal/non-preictal labels.
    """
    X = []
    y = []

    for edf_path, seizure_list in metadata.items():

        subject = os.path.basename(os.path.dirname(edf_path))
        edf_file = os.path.basename(edf_path)

        edf_full_path = os.path.join(
            base_dir,
            subject,
            edf_file
        )

        try:
            # Load EDF recording
            raw = mne.io.read_raw_edf(
                edf_full_path,
                preload=True,
                verbose=False
            )

            # Shape: (n_channels, n_samples)
            signal = raw.get_data()

            # Apply EEG preprocessing while preserving channels
            signal = preprocess_signal(signal)

            windows = []
            labels = []

            for seizure in seizure_list:

                if (
                    isinstance(seizure, (list, tuple))
                    and len(seizure) == 2
                ):
                    start_time, duration = seizure

                elif isinstance(seizure, (int, float)):
                    start_time = seizure
                    duration = 60

                else:
                    print(
                        f"[WARNING] Skipping {edf_file} due to "
                        f"invalid seizure annotation: {seizure}"
                    )
                    continue

                # Generate overlapping EEG windows
                current_windows = window_eeg_signal(
                    signal,
                    sfreq=256,
                    window_sec=window_sec,
                    step_sec=5
                )

                # Label windows relative to seizure onset
                current_labels = label_windows_with_preictal(
                    current_windows,
                    start_time,
                    preictal_sec=preictal_sec,
                    sampling_rate=256,
                    step_sec=5
                )

                if (
                    current_windows
                    and current_labels
                    and len(current_windows) == len(current_labels)
                ):
                    windows.extend(current_windows)
                    labels.extend(current_labels)

            if (
                windows
                and labels
                and len(windows) == len(labels)
            ):
                X.extend(windows)
                y.extend(labels)

            print(
                f"[INFO] {edf_file} -> "
                f"windows: {len(windows)}, "
                f"seizure events: {len(seizure_list)}"
            )

        except Exception as e:
            print(f"[ERROR] {edf_file}: {e}")

    if X:
        X = np.array(X)
    else:
        X = np.array([])

    if y:
        y = np.array(y)
    else:
        y = np.array([])

    return X, y


# ---------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------
if __name__ == "__main__":

    print("\n[INFO] Loading CHB-MIT seizure metadata...")

    # Dataset path can be supplied using the CHBMIT_DATA_DIR
    # environment variable.
    base_dir = os.getenv(
        "CHBMIT_DATA_DIR",
        "data/chb-mit"
    )

    seizure_metadata = load_chbmit_seizure_metadata(
        base_dir
    )

    if not seizure_metadata:
        print(
            "[ERROR] Seizure metadata is empty. "
            "Check CHBMIT_DATA_DIR or the dataset files."
        )
        raise SystemExit(1)

    X_chb, y_chb = load_chbmit_dataset(
        base_dir,
        seizure_metadata
    )

    print(
        "Data loaded successfully:",
        len(X_chb) if X_chb.size else 0,
        len(y_chb) if y_chb.size else 0
    )

    print(
        f"[DEBUG] X_chb shape: "
        f"{X_chb.shape if X_chb.size else 'Empty'}"
    )

    print(
        f"[DEBUG] y_chb shape: "
        f"{y_chb.shape if y_chb.size else 'Empty'}"
    )


    # -----------------------------------------------------------------
    # Mayo iEEG
    # -----------------------------------------------------------------
    print("\n[INFO] Loading Mayo iEEG signals...")

    X_sec, y_sec = load_mayo_dataset(
        max_files=12
    )

    print(
        f"[DEBUG] X_sec shape: "
        f"{X_sec.shape if X_sec.size else 'Empty'}"
    )

    print(
        f"[DEBUG] y_sec shape: "
        f"{y_sec.shape if y_sec.size else 'Empty'}"
    )


    # -----------------------------------------------------------------
    # Combine datasets
    # -----------------------------------------------------------------
    if X_chb.size == 0 and X_sec.size == 0:

        print(
            "[ERROR] Both datasets are empty. "
            "Check previous errors."
        )
        raise SystemExit(1)

    elif X_chb.size == 0:

        print(
            "[WARNING] CHB-MIT data are empty. "
            "Proceeding with Mayo data only."
        )

        X_combined = X_sec
        y_combined = y_sec

    elif X_sec.size == 0:

        print(
            "[WARNING] Mayo data are empty. "
            "Proceeding with CHB-MIT data only."
        )

        X_combined = X_chb
        y_combined = y_chb

    else:

        print("\n[INFO] Combining datasets...")

        # Use the common number of channels
        n_channels = min(
            X_chb.shape[1],
            X_sec.shape[1]
        )

        if n_channels == 0:
            print(
                "[ERROR] No compatible EEG channels found."
            )
            raise SystemExit(1)

        X_chb = X_chb[:, :n_channels, :]
        X_sec = X_sec[:, :n_channels, :]

        # Ensure both datasets use the same window length
        if X_chb.shape[2] != X_sec.shape[2]:

            target_length = min(
                X_chb.shape[2],
                X_sec.shape[2]
            )

            X_chb = X_chb[:, :, :target_length]
            X_sec = X_sec[:, :, :target_length]

        # Combine after harmonising dimensions
        X_combined = np.vstack(
            [X_chb, X_sec]
        )

        y_combined = np.concatenate(
            [y_chb, y_sec]
        )

    print(
        f"[DEBUG] X_combined shape: "
        f"{X_combined.shape}"
    )

    print(
        f"[DEBUG] y_combined shape: "
        f"{y_combined.shape}"
    )


    # -----------------------------------------------------------------
    # Validate combined data
    # -----------------------------------------------------------------
    if X_combined.size == 0:

        print(
            "[ERROR] Combined dataset is empty."
        )
        raise SystemExit(1)

    print(
        f"\n[INFO] Combined windows: {X_combined.shape}"
    )

    print(
        f"[INFO] Label distribution: "
        f"{np.bincount(y_combined.astype(int))}"
    )


    # -----------------------------------------------------------------
    # Feature extraction
    # -----------------------------------------------------------------
    print(
        "\n[INFO] Extracting statistical features..."
    )

    X_features = extract_features(
        X_combined
    )


    # -----------------------------------------------------------------
    # Majority-class downsampling
    # -----------------------------------------------------------------
    print(
        "\n[INFO] Downsampling majority class..."
    )

    y_combined_int = y_combined.astype(int)

    X_majority = X_features[
        y_combined_int == 0
    ]

    y_majority = y_combined_int[
        y_combined_int == 0
    ]

    X_minority = X_features[
        y_combined_int == 1
    ]

    y_minority = y_combined_int[
        y_combined_int == 1
    ]

    n_minority = len(y_minority)
    n_majority = len(y_majority)

    if n_minority > 0 and n_majority > 0:

        n_samples_down = min(
            5 * n_minority,
            n_majority
        )

    else:
        n_samples_down = 0


    if n_samples_down == 0:

        print(
            "[WARNING] Downsampling skipped because "
            "one of the classes is empty."
        )

        X_balanced = X_features
        y_balanced = y_combined_int

    else:

        X_majority_down, y_majority_down = (
            sklearn_resample(
                X_majority,
                y_majority,
                replace=False,
                n_samples=n_samples_down,
                random_state=42
            )
        )

        X_balanced = np.vstack(
            [
                X_majority_down,
                X_minority
            ]
        )

        y_balanced = np.concatenate(
            [
                y_majority_down,
                y_minority
            ]
        )


    # -----------------------------------------------------------------
    # Standardisation and SMOTE
    # -----------------------------------------------------------------
    print("\n[INFO] Applying SMOTE...")

    if X_balanced.size:

        scaler = StandardScaler()

        X_balanced = scaler.fit_transform(
            X_balanced
        )

        X_resampled, y_resampled = apply_smote(
            X_balanced,
            y_balanced
        )

        print(
            f"\n[INFO] Final dataset size after SMOTE: "
            f"{X_resampled.shape}"
        )

        print(
            f"[INFO] Final label distribution: "
            f"{np.bincount(y_resampled.astype(int))}"
        )

    else:

        print(
            "[WARNING] Balanced dataset is empty. "
            "Skipping SMOTE."
        )

        X_resampled = np.array([])
        y_resampled = np.array([])
    