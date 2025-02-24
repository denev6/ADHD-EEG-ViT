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
            data = dataset["data"]
            label = dataset["label"]
            self.assertEqual(
                data.size()[0],
                label.size()[0],
                f"{name} data and label size mismatch (expected: {label.size()[0]}, got {data.size()[0]})",
            )
            self.assertEqual(
                data.size()[0],
                self.META[f"{name}_size"],
                f"Unexpected {name} data size (expected: {self.META[f'{name}_size']}, got {data.size()[0]})",
            )

    def test_label_only_contains_0_and_1(self):
        for name, dataset in self.DATASETS.items():
            zeros = torch.sum(dataset["label"] == 0).item()
            ones = torch.sum(dataset["label"] == 1).item()
            self.assertEqual(
                zeros + ones,
                dataset["label"].size()[0],
                f"Label should only contain 0 and 1 ({name}-set)",
            )


if __name__ == "__main__":
    unittest.main()
