import os
import mne
import numpy as np

def load_chbmit_seizure_metadata(base_dir):
    """Load seizure metadata from summary files in the CHB-MIT dataset."""
    seizure_metadata = {}
    for subject in os.listdir(base_dir):
        subject_path = os.path.join(base_dir, subject)
        if os.path.isdir(subject_path):
            summary_file = os.path.join(subject_path, f'{subject}-summary.txt')
            if os.path.exists(summary_file):
                with open(summary_file, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        if 'File Name' in line or 'Seizure' in line:
                            continue
                        parts = line.strip().split()
                        if len(parts) >= 3 and parts[0].endswith('.edf'):
                            edf_file = parts[0]
                            if len(parts) > 2 and 'Seizure' in parts[2]:
                                start = int(parts[3])
                                duration = int(parts[4])
                                key = f"{subject}/{edf_file}"
                                if key not in seizure_metadata:
                                    seizure_metadata[key] = []
                                seizure_metadata[key].append((start, duration))
    return seizure_metadata

def load_chbmit_dataset(base_dir, metadata, window_sec=10, preictal_sec=300):
    """Load and process CHB-MIT EEG data into windows and labels."""
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