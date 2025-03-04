import unittest
from unittest.mock import patch
import posixpath

import torch.nn as nn
import torch.optim
from torch.utils.data import TensorDataset
from torch.amp import is_autocast_available

from .config import *
from .function import *
from .training import *


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
        metrics = evaluate(model, device(), dataloader)
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

        log_dict = {"C": 3, "D": None}
        wrong_log_dict = {"A": 5, "C": 5}
        config_dict = {"config": Config()}

        json_path = "dummy_log.json"
        try:
            logs = log_json(json_path, Config())
            self.assertIn("A", logs.keys(), "Dataclass attributes not included.")

            logs = log_json(json_path, **config_dict, c=Config())
            self.assertIsInstance(logs, dict, "Logs should be a dictionary.")
            self.assertNotIn("D", logs.keys(), "null value should be ignored.")

            with self.assertRaises(AssertionError):
                # Not a json format
                log_json("aaa.yaml", **config_dict, **log_dict)

            with self.assertRaises(TypeError):
                # Duplicate keyword arguments
                log_json(json_path, **log_dict, **wrong_log_dict)

            with self.assertRaises(KeyError):
                # Duplicate keys in dataclass
                log_json(json_path, Config(), **wrong_log_dict)

        finally:
            if os.path.exists(json_path):
                os.remove(json_path)


class TestConfig(unittest.TestCase):
    def test_config(self):
        name = "test_config"
        config = Config(
            name=name,
            batch=4,
            epochs=5,
            lr=0.01,
        )
        self.assertEqual(config.name, name, "Config name mismatch")
        self.assertTrue(hasattr(config, "id"), "Config.id not initialized")

        config = Config(
            name=name,
            batch=4,
            epochs=5,
            lr=0.01,
        )
        config.add(optimizer=torch.optim.Adam)
        self.assertTrue(hasattr(config, "optimizer"), "Extra attributes not saved")

        with self.assertRaises(KeyError):
            # Duplicate keys in dataclass
            config = Config(
                name=name,
                batch=4,
                epochs=5,
                lr=0.01,
            )
            config.add(id=1)

        with self.assertRaises(TypeError):
            # Unexpected Arguments
            Config(name=name, batch=4, epochs=5, lr=0.01, something=0)


class TestTraining(unittest.TestCase):

    def _get_2d_model(self):
        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.fc = nn.Linear(2, 2)

            def forward(self, x):
                return self.fc(x)

        return Model

    def test_internal_get_2d_tensor_dataset(self):
        dataset = self._get_2d_tensor_dataset()
        self.assertIsInstance(dataset, TensorDataset)

        data_tensor, label_tensor = dataset.tensors
        self.assertIsInstance(data_tensor, torch.FloatTensor)
        self.assertIsInstance(label_tensor, torch.LongTensor)
        self.assertEqual(
            data_tensor.size(0),
            label_tensor.size(0),
            "Data and label tensors have different shapes",
        )
        self.assertEqual(
            data_tensor.size(1), 2, "Data tensor has incorrect number of features"
        )

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
        initial_lr = 300
        warmup_steps = 3
        losses = [5, 4, 3, 5, 6]
        expected_lr = [100, 200, 300, 300, 30, 3]
        model = self._get_2d_model()()

        optimizer = torch.optim.SGD(model.parameters(), initial_lr)
        scheduler = WarmupScheduler(
            optimizer, initial_lr, warmup_steps=warmup_steps, decay_factor=0.1
        )

        for i, loss in enumerate(losses):
            current_lr = scheduler.get_lr()[0]
            self.assertEqual(
                expected_lr[i], current_lr, "Learning rate not updated as expected"
            )
            # Assume that the model is trained here.
            scheduler.step(loss)

        with self.assertRaises(AssertionError):
            WarmupScheduler(optimizer, lr=1e-5, min_lr=1e-3)

        with self.assertRaises(AssertionError):
            WarmupScheduler(optimizer, initial_lr, decay_factor=10)

    def _get_2d_tensor_dataset(self):
        inputs = torch.tensor([[0, 1], [0, 1], [1, 0], [1, 0], [1, 0]])
        true_labels = torch.tensor([1, 0, 1, 0, 0])
        dataset = TensorDataset(inputs.float(), true_labels.long())
        return dataset

    def test_training(self):
        dataset = self._get_2d_tensor_dataset()
        train_dataloader = DataLoader(dataset, batch_size=5)
        val_dataloader = DataLoader(dataset, batch_size=5)

        model = self._get_2d_model()()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        model_path = "dummy_model.pt"
        epochs = 1

        try:
            check_point = train(
                model,
                device(),
                model_path,
                optimizer,
                criterion,
                epochs,
                train_dataloader,
                val_dataloader,
            )
            self.assertIsInstance(check_point, int)
        finally:
            if os.path.exists(model_path):
                os.remove(model_path)

    def test_training_with_torch_scheduler(self):
        dataset = self._get_2d_tensor_dataset()
        train_dataloader = DataLoader(dataset, batch_size=5)
        val_dataloader = DataLoader(dataset, batch_size=5)

        model = self._get_2d_model()()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
        model_path = "dummy_model.pt"
        epochs = 1

        try:
            check_point = train(
                model,
                device(),
                model_path,
                optimizer,
                criterion,
                epochs,
                train_dataloader,
                val_dataloader,
                scheduler=scheduler,
            )
            self.assertIsInstance(check_point, int)
        finally:
            if os.path.exists(model_path):
                os.remove(model_path)

    @unittest.skipIf(not is_autocast_available(str(device())), "AMP is not available.")
    def test_training_in_fp16(self):
        dataset = self._get_2d_tensor_dataset()
        train_dataloader = DataLoader(dataset, batch_size=5)
        val_dataloader = DataLoader(dataset, batch_size=5)

        model = self._get_2d_model()()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        model_path = "dummy_model.pt"
        epochs = 1

        try:
            check_point = train(
                model,
                device(),
                model_path,
                optimizer,
                criterion,
                epochs,
                train_dataloader,
                val_dataloader,
                enable_fp16=True,
            )
            self.assertIsInstance(check_point, int)
        finally:
            if os.path.exists(model_path):
                os.remove(model_path)

    def test_training__with_kfold(self):
        dataset = self._get_2d_tensor_dataset()
        model = self._get_2d_model()
        optimizer = torch.optim.SGD
        optimizer_params = {"lr": 1e-3}
        criterion = nn.CrossEntropyLoss()
        model_path = "dummy_model.pt"
        epochs = 1
        k_folds = 3

        try:
            check_point, best_model = train_with_kfold(
                k_folds,
                model,
                device(),
                model_path,
                optimizer,
                criterion,
                epochs,
                dataset,
                batch=5,
                optimizer_params=optimizer_params,
            )
            self.assertIsInstance(check_point, int)
        finally:
            for i in range(k_folds):
                model_path = f"dummy_model_{i+1}.pt"
                if os.path.exists(model_path):
                    os.remove(model_path)


if __name__ == "__main__":
    unittest.main()
