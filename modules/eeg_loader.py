# modules/load_data.py
import os
import mne

def load_edf_file(file_path):
    print(f" Reading file with MNE: {file_path}")
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    return raw

def list_edf_files(directory):
    return [f for f in os.listdir(directory) if f.endswith('.edf')]


