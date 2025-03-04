import numpy as np
import matplotlib.pyplot as plt

check_point = 50
epochs = 50
title = "Loss"

logs = """
Epoch 1, Train-Loss: 0.76217,  Val-Loss: 0.69103
    ...
Epoch 50, Train-Loss: 0.10157,  Val-Loss: 0.29716
"""


logs = logs.split("\n")[1:-1]
assert len(logs) == epochs, "Number of logs does not match number of epochs."
train_val_loss = [str(log).split(",")[1:] for log in logs]
trains = []
vals = []

for train, val in train_val_loss:
    trains.append(float(train.split(":")[-1].strip()))
    vals.append(float(val.split(":")[-1].strip()))

x = np.arange(0, len(trains))

plt.figure(figsize=(8, 6))
plt.plot(x, trains, "-", color="blue", label="Training Loss")
plt.plot(x[:check_point], vals[:check_point], "-", color="red", label="Validation Loss")
plt.plot(x[check_point:], vals[check_point:], "--", color="red")
plt.scatter(x[check_point - 1], vals[check_point - 1], color="red", label="Check Point")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(title)
plt.legend()
plt.show()
