import os
import unittest
import torch
import torch.nn as nn
import torch.optim

from training import EarlyStopping, WarmupScheduler


class EmptyModel(nn.Module):
    def __init__(self):
        super(EmptyModel, self).__init__()
        self.fc = nn.Linear(1, 1)

    def forward(self, x):
        return self.fc(x)


class TestUtils(unittest.TestCase):
    def test_early_stopping(self):
        model_path = "dummy_test_early_stopping.pt"
        early_stopping = EarlyStopping(patience=2, path_to_save=model_path)
        losses = [3, 2, 1, 2, 3, 4, 5]
        expected_checkpoint = 3
        expected_last_epoch = 5
        model = EmptyModel()

        for epoch, loss in enumerate(losses, start=1):
            if early_stopping.should_stop(loss, model, epoch):
                break

        self.assertEqual(
            early_stopping.check_point,
            expected_checkpoint,
            f"Different checkpoints (expected: {expected_checkpoint}, got {early_stopping.check_point})",
        )
        self.assertEqual(
            epoch,
            expected_last_epoch,
            f"Stop earlier than expected (expected: {expected_last_epoch}, got {epoch})",
        )
        self.assertTrue(os.path.exists(model_path), "Model checkpoint not saved")
        os.remove(model_path)  # Remove dummy checkpoint

    def test_warmup_scheduler(self):
        initial_lr = 300
        warmup_steps = 3
        losses = [5, 4, 3, 5, 6]
        expected_lr = [100, 200, 300, 300, 30, 3]

        optimizer = torch.optim.SGD(EmptyModel().parameters(), initial_lr)
        scheduler = WarmupScheduler(optimizer, initial_lr, warmup_steps=warmup_steps)

        for i, loss in enumerate(losses):
            current_lr = scheduler.get_lr()[0]
            self.assertEqual(
                current_lr, expected_lr[i], "Learning rate not updated as expected"
            )
            # Assume that the model is trained here.
            scheduler.step(loss)


if __name__ == "__main__":
    unittest.main()
