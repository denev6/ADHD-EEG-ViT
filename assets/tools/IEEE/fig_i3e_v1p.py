import numpy as np
import matplotlib.pyplot as plt
import scipy.io

mat_file = "../../backup_datasets/IEEE/ADHD_part1/v1p.mat"
data = scipy.io.loadmat(mat_file)
EEG_signal = data["v1p"].astype(np.int32)  # Shape (12258, 19)
time_vector = np.arange(1, 12259)

plt.figure(figsize=(12, 8))
offset = 3000

for ch in range(EEG_signal.shape[1]):
    plt.plot(time_vector, EEG_signal[:, ch] + (ch + 1) * offset, "k")

plt.title("EEG Channels (v1p)")
plt.yticks(offset * (np.arange(19) + 1))

ax = plt.gca()
ax.set_yticklabels([f"Ch {i+1}" for i in range(19)])

plt.grid(True)
plt.show()
