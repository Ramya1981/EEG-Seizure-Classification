import os
import numpy as np
from scipy.signal import resample as scipy_resample
from sklearn.utils import resample as sklearn_resample
from sklearn.preprocessing import StandardScaler
from modules.metadataloader import load_chbmit_seizure_metadata, load_chbmit_dataset
from modules.windowing import window_eeg_signal, label_windows_with_preictal
from modules.eeg_preprocess import preprocess_signal
from modules.smote_utils import apply_smote
from modules.load_mayo import load_mayo_dataset
import mne

# Feature extraction for SMOTE
def extract_features(windows):
    features = []
    for window in windows:
        mean = np.mean(window, axis=1)
        var = np.var(window, axis=1)
        feat = np.concatenate([mean, var])
        features.append(feat)
    return np.array(features)

# Load CHB-MIT dataset (already provided and corrected)
def load_chbmit_dataset(base_dir, metadata, window_sec=10, preictal_sec=300):
    X, y = [], []
    for edf_path, seizure_list in metadata.items():
        subject = os.path.basename(os.path.dirname(edf_path))
        edf_file = os.path.basename(edf_path)
        edf_full_path = os.path.join(base_dir, subject, edf_file)

        try:
            raw = mne.io.read_raw_edf(edf_full_path, preload=True, verbose=False)
            signal = raw.get_data()
            signal = np.mean(signal, axis=0)  # Average across channels
            signal = preprocess_signal(signal)

            windows = []
            labels = []
            for seizure in seizure_list:
                if isinstance(seizure, (list, tuple)) and len(seizure) == 2:
                    start_time, duration = seizure
                elif isinstance(seizure, (int, float)):
                    start_time = seizure
                    duration = 60  # Default assumption
                else:
                    print(f"[WARNING] Skipping {edf_file} due to invalid seizure annotation: {seizure}")
                    continue

                current_windows = window_eeg_signal(signal, 256, window_sec, 5)
                current_labels = label_windows_with_preictal(
                    current_windows, start_time, preictal_sec=preictal_sec, sampling_rate=256, step_sec=5
                )

                if current_windows and current_labels and len(current_windows) == len(current_labels):
                    windows.extend(current_windows)
                    labels.extend(current_labels)

            if windows and labels and len(windows) == len(labels):
                X.extend(windows)
                y.extend(labels)

            print(f"[INFO] {edf_file} → windows: {len(windows)}, seizure events: {len(seizure_list)}")

        except Exception as e:
            print(f"[ERROR] {edf_file}: {e}")

    return np.array(X) if X and len(X) > 0 else np.array([]), np.array(y) if y and len(y) > 0 else np.array([])

# Main Execution
if __name__ == "__main__":
    print("\n[INFO] Loading CHB-MIT seizure metadata...")
    base_dir = "C:/Users/Ramya Sundaram/Downloads/chb-mit-scalp-eeg-database-1.0.0/chb-mit-scalp-eeg-database-1.0.0"
    seizure_metadata = load_chbmit_seizure_metadata(base_dir)
    if not seizure_metadata:
        print("[ERROR] Seizure metadata is empty. Check base_dir or summary files.")
        exit(1)

    X_chb, y_chb = load_chbmit_dataset(base_dir, seizure_metadata)
    print("Data loaded successfully:", len(X_chb) if X_chb.size else 0, len(y_chb) if y_chb.size else 0)
    print(f"[DEBUG] X_chb shape: {X_chb.shape if X_chb.size else 'Empty'}")
    print(f"[DEBUG] y_chb shape: {y_chb.shape if y_chb.size else 'Empty'}")

    print("\n[INFO] Loading Mayo iEEG signals...")
    X_sec, y_sec = load_mayo_dataset(max_files=12)
    print(f"[DEBUG] X_sec shape: {X_sec.shape if X_sec.size else 'Empty'}")
    print(f"[DEBUG] y_sec shape: {y_sec.shape if y_sec.size else 'Empty'}")

    if X_chb.size == 0 and X_sec.size == 0:
        print("[ERROR] Both datasets are empty. Check previous errors.")
        exit(1)
    elif X_chb.size == 0:
        print("[WARNING] X_chb is empty. Proceeding with X_sec only.")
        X_combined = X_sec
        y_combined = y_sec
    elif X_sec.size == 0:
        print("[WARNING] X_sec is empty. Proceeding with X_chb only.")
        X_combined = X_chb
        y_combined = y_chb
    else:
        print("\n[INFO] Combining datasets...")
        n_channels = min(X_chb.shape[1], X_sec.shape[1]) if X_chb.size and X_sec.size else 0
        if n_channels == 0:
            print("[ERROR] Incompatible channel dimensions or empty datasets.")
            exit(1)
        X_chb = X_chb[:, :n_channels, :] if X_chb.size else np.array([])
        X_sec = X_sec[:, :n_channels, :] if X_sec.size else np.array([])
        if X_chb.size and X_sec.size and X_chb.shape[2] != X_sec.shape[2]:
            target_length = min(X_chb.shape[2], X_sec.shape[2])
            X_chb = X_chb[:, :, :target_length]
            X_sec = X_sec[:, :, :target_length]
        elif X_chb.size == 0 or X_sec.size == 0:
            X_combined = X_chb if X_chb.size else X_sec
            y_combined = y_chb if y_chb.size else y_sec
        else:
            X_combined = np.vstack([X_chb, X_sec])
            y_combined = np.concatenate([y_chb, y_sec])

        print(f"[DEBUG] X_combined shape: {X_combined.shape if X_combined.size else 'Empty'}")
        print(f"[DEBUG] y_combined shape: {y_combined.shape if y_combined.size else 'Empty'}")

    if X_combined.size == 0:
        print("[ERROR] X_combined is empty after processing. Exiting.")
        exit(1)

    print(f"\n[INFO] Combined windows: {X_combined.shape} | Labels: {np.bincount(y_combined.astype(int))}")

    print("\n[INFO] Extracting features for SMOTE...")
    X_features = extract_features(X_combined)

    print("\n[INFO] Downsampling majority class...")
    y_combined_int = y_combined.astype(int)
    X_majority = X_features[y_combined_int == 0]
    y_majority = y_combined_int[y_combined_int == 0]
    X_minority = X_features[y_combined_int == 1]
    y_minority = y_combined_int[y_combined_int == 1]

    n_minority = len(y_minority)
    n_majority = len(y_majority)
    n_samples_down = min(5 * n_minority, n_majority) if n_minority > 0 and n_majority > 0 else 0

    if n_samples_down == 0:
        print("[WARNING] No downsampling performed due to insufficient minority or majority samples.")
        X_balanced = X_features
        y_balanced = y_combined_int
    else:
        X_majority_down, y_majority_down = sklearn_resample(
            X_majority, y_majority,
            replace=False,
            n_samples=n_samples_down,
            random_state=42
        )
        X_balanced = np.vstack([X_majority_down, X_minority])
        y_balanced = np.concatenate([y_majority_down, y_minority])

    print(f"X_chb shape: {X_chb.shape if X_chb.size else 'Empty'}")
    print(f"X_sec shape: {X_sec.shape if X_sec.size else 'Empty'}")

    print("\n[INFO] Applying SMOTE...")
    scaler = StandardScaler()
    X_balanced = scaler.fit_transform(X_balanced) if X_balanced.size else np.array([])
    if X_balanced.size:
        X_resampled, y_resampled = apply_smote(X_balanced, y_balanced)
        print(f"\n[INFO] Final dataset size after SMOTE: {X_resampled.shape} | Labels: {np.bincount(y_resampled.astype(int))}")
    else:
        print("[WARNING] X_balanced is empty. Skipping SMOTE.")
        X_resampled, y_resampled = np.array([]), np.array([])

    