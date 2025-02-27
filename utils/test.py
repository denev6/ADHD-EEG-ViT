import unittest
from unittest.mock import patch
import posixpath
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader, TensorDataset

from .function import *
from .training import EarlyStopping, WarmupScheduler


class TestFunction(unittest.TestCase):

    @patch("os.path.join", side_effect=posixpath.join)
    def test_join_drive_path(self, mock_join):
        # 'join_drive_path' is designed to run on Linux.
        self.assertEqual(
            "/content/drive/MyDrive/aaa/bbb.py",
            join_drive_path("aaa", "bbb.py"),
            "Joining drive path failed.",
        )

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is not available.")
    def test_device_returns_cuda_if_available(self):
        self.assertEqual(
            torch.device("cuda"), device(force_cuda=True), "Device should be 'cuda'."
        )

    def test_evaluate(self):
        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.identity = nn.Identity()

            def forward(self, x):
                return self.identity(x)

        inputs = torch.tensor([[0, 1], [0, 1], [1, 0], [1, 0], [1, 0]])
        true_labels = torch.tensor([1, 0, 1, 0, 0])
        expected_metrics = {"accuracy": 0.6, "f1-score": 0.5, "recall": 0.5, "auc": 0.5}
        delta = 0.01

        dataset = TensorDataset(inputs, true_labels)
        dataloader = DataLoader(dataset, batch_size=5)
        model = Model()
        metrics = evaluate(model, torch.device("cpu"), dataloader)
        self.assertAlmostEqual(
            expected_metrics["accuracy"], metrics["accuracy"], delta=delta
        )
        self.assertAlmostEqual(
            expected_metrics["f1-score"], metrics["f1-score"], delta=delta
        )
        self.assertAlmostEqual(
            expected_metrics["recall"], metrics["recall"], delta=delta
        )
        self.assertAlmostEqual(
            expected_metrics["auc"], float(metrics["auc"]), delta=0.1
        )

    def test_log_json(self):
        @dataclass
        class Config:
            A: int = 1

        class WrongConfig:
            pass

        log_dict = {"C": 3}
        wrong_log_dict = {"A": 3}

        json_path = "dummy_log.json"
        try:
            logs = log_json(json_path, Config(), **log_dict)
            self.assertIsInstance(logs, dict, "Logs should be a dictionary.")

            with self.assertRaises(AssertionError):
                # Not a json format
                log_json("aaa.yaml", Config(), **log_dict)

            with self.assertRaises(ValueError):
                # Not a dataclass
                log_json(json_path, WrongConfig(), **log_dict)

            with self.assertRaises(KeyError):
                # Duplicate keys
                log_json(json_path, Config(), **wrong_log_dict)

        finally:
            if os.path.exists(json_path):
                os.remove(json_path)


class TestUtils(unittest.TestCase):
    def test_early_stopping(self):
        class EmptyModel(nn.Module):
            pass

        model_path = "dummy_test_early_stopping.pt"
        early_stopping = EarlyStopping(patience=2, path_to_save=model_path)
        losses = [3, 2, 1, 2, 3, 4, 5]
        expected_checkpoint = 3
        expected_last_epoch = 5
        model = EmptyModel()

        try:
            for epoch, loss in enumerate(losses, start=1):
                if early_stopping.should_stop(loss, model, epoch):
                    break

            self.assertEqual(
                expected_checkpoint,
                early_stopping.check_point,
                f"Different checkpoints",
            )
            self.assertEqual(
                expected_last_epoch,
                epoch,
                f"Stop earlier than expected",
            )
            self.assertTrue(os.path.exists(model_path), "Model checkpoint not saved")
        finally:
            if os.path.exists(model_path):
                # Remove dummy checkpoint made for the test
                os.remove(model_path)

    def test_warmup_scheduler(self):
        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.fc = nn.Linear(1, 1)

            def forward(self, x):
                return self.fc(x)

        initial_lr = 300
        warmup_steps = 3
        losses = [5, 4, 3, 5, 6]
        expected_lr = [100, 200, 300, 300, 30, 3]

        optimizer = torch.optim.SGD(Model().parameters(), initial_lr)
        scheduler = WarmupScheduler(optimizer, initial_lr, warmup_steps=warmup_steps)

        for i, loss in enumerate(losses):
            current_lr = scheduler.get_lr()[0]
            self.assertEqual(
                expected_lr[i], current_lr, "Learning rate not updated as expected"
            )
            # Assume that the model is trained here.
            scheduler.step(loss)


if __name__ == "__main__":
    unittest.main()
