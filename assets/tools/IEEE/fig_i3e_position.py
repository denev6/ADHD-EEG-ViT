import pandas as pd
import matplotlib.pyplot as plt
import mne

# Load the .ced file
ced_file = "../../backup_datasets/IEEE/Standard-10-20-Cap19new copy.ced"
df = pd.read_csv(ced_file, sep="\t")

electrode_data = df[["labels", "X", "Y", "Z"]]

ch_pos = {
    row["labels"]: [-row["Y"], row["X"], row["Z"]]
    for _, row in electrode_data.iterrows()
}

montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame="head")
montage.plot(sphere=1.3)
plt.show()
