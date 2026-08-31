# modules/eeg_preprocess.py
from scipy.signal import butter, filtfilt

def preprocess_signal(signal, sfreq=256):
    """
    Preprocess EEG signal (e.g., bandpass filter).
    Input: signal (n_channels, n_samples)
    Output: filtered signal (n_channels, n_samples)
    """
    b, a = butter(4, [0.5, 70], btype='band', fs=sfreq)
    return filtfilt(b, a, signal, axis=-1)