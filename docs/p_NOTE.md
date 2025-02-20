# Private NOTE

- OSF: MATLAB 5.0 MAT-file

Loading v7.3 .mat Files (HDF5-Based)

```python
import h5py

file_path = "data.mat"
with h5py.File(file_path, 'r') as f:
    data = f["variable_name"][:]  # Replace with actual variable name
```

Loading v7.0 and Older .mat Files

```python
from scipy.io import loadmat

data = loadmat("data.mat")
print(data.keys())
```
