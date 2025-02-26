from dataclasses import dataclass
import torch
from torch.utils.data import Dataset


class EEGDataset(Dataset):
    def __init__(self, file_path):
        self.dataset = torch.load(file_path)
        self.eeg = self.dataset["data"]
        self.labels = self.dataset["label"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # eeg: FloatTensor to match weights and bias
        # labels: LongTensor for loss computation
        return self.eeg[idx].float(), self.labels[idx].long()

    @staticmethod
    def decode(label: int):
        return ["Control", "ADHD"][label]


@dataclass(frozen=True)
class IEEEData:
    """Constants for IEEE dataset."""

    tag = "IEEE_23"
    train = "ieee_train.pt"
    test = "ieee_test.pt"
    val = "ieee_val.pt"
    channels = 19
    length = 2560
    num_classes = 2
