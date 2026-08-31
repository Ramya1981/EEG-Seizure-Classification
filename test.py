import scipy.io
import os
import numpy as np
# Test on 1 file from folder '3' (pathological)
file_path = r"C:/Users/Ramya Sundaram/Downloads/archive/DatasetMayo/DatasetMayo/3"
mat_file = os.listdir(file_path)[0]
full_path = os.path.join(file_path, mat_file)

print(f"Inspecting file: {full_path}")

mat = scipy.io.loadmat(full_path)

# Print all keys and shapes
for key in mat:
    print(f"Key: {key}")
    if isinstance(mat[key], np.ndarray):
        print(f"  Shape: {mat[key].shape}")
