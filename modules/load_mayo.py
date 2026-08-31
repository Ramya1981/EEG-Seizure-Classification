# modules/load_mayo.py
import numpy as np
import h5py
from scipy.signal import resample
import os
from modules.windowing import window_eeg_signal

def load_mayo_dataset(max_files=2):
    """
    Load Mayo iEEG dataset (400 Hz) from subdirectories 2 and 3, resample to 256 Hz.
    Uses directory names for labeling (2: Physiological, 3: Pathological).
    Output: X (n_windows, n_channels, window_size), y (n_windows,)
    """
    X, y = [], []
    mayo_dir = os.getenv("MAYO_DATA_DIR", "data/mayo")
    if not os.path.exists(mayo_dir):
        print(f"[ERROR] Mayo directory not found: {mayo_dir}")
        return np.array([]), np.array([])

    # Recursively find .mat files in directories 2 and 3
    mat_files = []
    for root, dirs, files in os.walk(mayo_dir):
        dir_name = os.path.basename(root)
        if dir_name in ['2', '3']:  # Only process Physiological and Pathological
            for file in files:
                if file.endswith('.mat'):
                    mat_files.append(os.path.join(root, file))
    mat_files = mat_files[:max_files]  # Limit to max_files
    print(f"[DEBUG] Mayo files found: {mat_files}")

    for file in mat_files:
        with h5py.File(file, 'r') as mat:
            print(f"[DEBUG] {file} - Keys: {list(mat.keys())}")
            # Adjust based on actual data structure (e.g., 'data' or another key)
            data = mat['data'][:] if 'data' in mat else mat[list(mat.keys())[0]][:]  # Fallback to first key
            if data.size == 0:
                print(f"[WARNING] {file} - No valid data found, skipping")
                continue
            orig_sfreq = 400
            target_sfreq = 256
            n_samples_new = int(data.shape[1] * target_sfreq / orig_sfreq)
            data = resample(data, n_samples_new, axis=1)
            print(f"[DEBUG] {file} - Resampled data shape: {data.shape}")

            windows = window_eeg_signal(data, target_sfreq, window_sec=10, step_sec=5)
            # Assign label based on directory name
            label = int(os.path.basename(os.path.dirname(file)))
            labels = [label] * len(windows)
            print(f"[DEBUG] {file} - Windows created: {len(windows)}, Label: {label}")
            X.extend(windows)
            y.extend(labels)

    return np.array(X), np.array(y)