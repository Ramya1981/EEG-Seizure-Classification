import os
from collections import defaultdict

def parse_chbmit_summary(summary_path):
    seizure_files = defaultdict(list)
    nonseizure_files = defaultdict(list)

    current_file = ""
    current_patient = ""

    with open(summary_path, 'r') as f:
        for line in f:
            line = line.strip()

            if line.startswith("File Name:"):
                current_file = line.split(":")[1].strip()
                current_patient = current_file.split("_")[0]
            elif line.startswith("Number of Seizures in File:"):
                seizure_count = int(line.split(":")[1].strip())
                if seizure_count > 0:
                    seizure_files[current_patient].append(current_file)
                else:
                    nonseizure_files[current_patient].append(current_file)

    return seizure_files, nonseizure_files
