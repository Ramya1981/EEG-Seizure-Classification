import mne
import matplotlib.pyplot as plt

# Load the EDF file
edf_file = "C:/Users/Ramya Sundaram/Downloads/chb-mit-scalp-eeg-database-1.0.0/chb-mit-scalp-eeg-database-1.0.0/chb01/chb01_03.edf"  # Replace with your .edf file path
raw = mne.io.read_raw_edf(edf_file, preload=True, verbose=False)

# Basic info
print("\n--- EEG File Info ---")
print(raw.info)
print("\nAvailable Channels:", raw.ch_names)
print("Data Shape (channels × time points):", raw.get_data().shape)
print("Sampling frequency:", raw.info['sfreq'])

# Plot raw signals (time series)
raw.plot(n_channels=10, scalings='auto', title='Raw EEG Signal (First 10 Channels)', show=True, block=True)

# Plot Power Spectral Density (PSD) for all channels
raw.plot_psd(fmax=60, average=True)
plt.title("Power Spectral Density (0–60 Hz)")
plt.show()

# Show channel-wise statistics
import numpy as np
data = raw.get_data()
print("\n--- Channel-wise Statistics ---")
for i, ch_name in enumerate(raw.ch_names):
    ch_data = data[i]
    print(f"{ch_name}: mean={np.mean(ch_data):.3f}, std={np.std(ch_data):.3f}, min={np.min(ch_data):.3f}, max={np.max(ch_data):.3f}")

# Optional: plot specific channel
channel_to_plot = "Fp1-F7"  # Change to a valid channel name in your EDF
if channel_to_plot in raw.ch_names:
    raw.plot(picks=[channel_to_plot], title=f"EEG Channel: {channel_to_plot}", show=True, block=True)

# Plot duration in seconds
duration = raw.times[-1]
print(f"\nRecording Duration: {duration:.2f} seconds")

# Optional: set EEG montage (e.g., standard 10-20 layout)
try:
    raw.set_montage('standard_1020')
    raw.plot_sensors(show_names=True)
except Exception as e:
    print("Montage plotting skipped:", e)
