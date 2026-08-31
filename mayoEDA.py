import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

# File path
mat_file_path = r"C:\Users\Ramya Sundaram\Downloads\archive\DatasetMayo\DatasetMayo\3\p000092069.mat"

# Load .mat file
mat_file = h5py.File(mat_file_path, 'r')
print("Top-level keys in .mat file:", list(mat_file.keys()))

# Access 'data'
data = mat_file['data'][:]
print("Shape of 'data':", data.shape)
print("Data type:", data.dtype)

# Flatten the signal
signal = data[:, 0]

# Summary stats
print("\n--- EEG Signal Statistics ---")
print(f"Mean: {np.mean(signal):.4f}")
print(f"Std Dev: {np.std(signal):.4f}")
print(f"Min: {np.min(signal):.4f}")
print(f"Max: {np.max(signal):.4f}")

# Time series plot
plt.figure(figsize=(10, 4))
plt.plot(signal, color='darkblue')
plt.title("EEG Signal")
plt.xlabel("Time Points")
plt.ylabel("Amplitude (uV)")
plt.grid(True)
plt.tight_layout()
plt.show()

# PSD Plot
fs = 256  # Sampling rate (adjust if known)
f, psd = welch(signal, fs=fs, nperseg=1024)
plt.semilogy(f, psd)
plt.title("Power Spectral Density")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power/Frequency (dB/Hz)")
plt.xlim([0, 60])
plt.grid(True)
plt.tight_layout()
plt.show()

mat_file.close()

