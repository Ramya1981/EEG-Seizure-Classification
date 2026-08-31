import numpy as np

def window_eeg_signal(data, sfreq, window_sec=10, step_sec=5):
    n_samples = data.shape[0]  # Assuming (n_samples, n_channels)
    window_samples = int(window_sec * sfreq)
    step_samples = int(step_sec * sfreq)
    
    if n_samples < window_samples:
        print(f"[WARNING] Data length {n_samples/sfreq} sec too short for window {window_sec} sec. Using full data.")
        window_samples = n_samples  # Use full data as one window
        step_samples = n_samples    # No stepping
    
    windows = []
    for start in range(0, n_samples - window_samples + 1, step_samples):
        end = start + window_samples
        windows.append(data[start:end])
    print(f"[DEBUG] Created {len(windows)} windows")
    return windows

def label_windows_with_preictal(windows, metadata, sfreq, preictal_sec=30):
    labels = []
    for i, window in enumerate(windows):
        start_time = i * (window.shape[0] / sfreq)
        end_time = start_time + (window.shape[0] / sfreq)
        label = 0  # Default non-seizure
        for seizure_start, duration in metadata.get('seizures', []):
            seizure_end = seizure_start + duration
            if (seizure_end - preictal_sec <= end_time and start_time <= seizure_end):
                label = 2  # Seizure or preictal
                break
        labels.append(label)
    print(f"[DEBUG] Labeled {len(labels)} windows")
    return labels