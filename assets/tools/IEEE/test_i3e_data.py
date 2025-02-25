import os
import unittest
import json
import torch


class MyTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Load metadata and datasets."""
        data_dir = os.path.abspath("../../IEEE")

        with open(os.path.join(data_dir, "metadata.json"), "r") as file:
            cls.META = json.load(file)

        trainset = torch.load(os.path.join(data_dir, "train.pt"), weights_only=True)
        valset = torch.load(os.path.join(data_dir, "val.pt"), weights_only=True)
        testset = torch.load(os.path.join(data_dir, "test.pt"), weights_only=True)
        cls.DATASETS = {
            "train": trainset,
            "val": valset,
            "test": testset,
        }

    def test_data_size(self):
        for name, dataset in self.DATASETS.items():
            data_size = list(dataset["data"].size())
            label_size = list(dataset["label"].size())
            self.assertEqual(
                data_size[:1],
                label_size,
                f"{name} label size mismatch (expected: {label_size}, got {data_size[:1]})",
            )
            self.assertEqual(
                data_size,
                self.META[f"{name}_size"],
                f"{name} data size mismatch (expected: {self.META[f'{name}_size']}, got {data_size})",
            )

    def test_label_only_contains_0_and_1(self):
        for name, dataset in self.DATASETS.items():
            labels = dataset["label"]
            zeros = torch.sum(labels == 0).item()
            ones = torch.sum(labels == 1).item()
            self.assertEqual(
                zeros + ones,
                labels.size()[0],
                f"Label should only contain 0 and 1 ({name}-set)",
            )


if __name__ == "__main__":
    unittest.main()
