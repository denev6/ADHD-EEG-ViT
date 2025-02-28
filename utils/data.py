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
class IEEEDataConfig:
    """Constants for IEEE dataset."""

    tag: str = "IEEE_23"
    train: str = "ieee_train.pt"
    test: str = "ieee_test.pt"
    val: str = "ieee_val.pt"
    channels: int = 19
    length: int = 9250
    num_classes: int = 2
