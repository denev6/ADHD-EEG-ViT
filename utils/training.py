import torch
from tqdm.auto import tqdm, trange


class EarlyStopping(object):
    """Stop training when loss does not decrease"""

    def __init__(self, patience: int, path_to_save: str):
        self._min_loss = float("inf")
        self._patience = patience
        self._path = path_to_save
        self.__check_point = None
        self.__counter = 0

    def should_stop(self, loss: float, model: torch.nn.Module, epoch: int) -> bool:
        """Check if training should stop and save the check point if needed.

        :param loss: Current validation loss.
        :param model: Model to save (it will compare the model with prior saved model and save if better).
        :param epoch: current epoch (will be used as check point if needed).
        :return: True if training should stop, False otherwise.
        """
        if loss < self._min_loss:
            self._min_loss = loss
            self.__counter = 0
            self.__check_point = epoch
            torch.save(model.state_dict(), self._path)
        elif loss > self._min_loss:
            self.__counter += 1
            if self.__counter == self._patience:
                return True
        return False

    def load(self, weights_only=True):
        """Load best model weights"""
        return torch.load(self._path, weights_only=weights_only)

    @property
    def check_point(self):
        """Return check point index

        :return: check point index
        """
        if self.__check_point is None:
            raise ValueError("No check point is saved!")
        return self.__check_point


class WarmupScheduler(object):
    """Warmup learning rate and dynamically adjusts learning rate based on validation loss.

    When the loss increases, the learning rate will be divided by decay_factor.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        initial_lr: float,
        min_lr=1e-6,
        warmup_steps=10,
        decay_factor=10,
    ):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.decay_factor = decay_factor

        assert self.warmup_steps > 0, "Warmup steps must be greater than 0"
        assert self.decay_factor > 1, "Decay factor must be greater than 1"

        self.global_step = 1
        self.best_loss = float("inf")

        # Initialize learning rates
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.initial_lr * (self.global_step / self.warmup_steps)

    def step(self, loss: float):
        """Update learning rate based on current loss.

        :param loss: Current validation loss.
        """
        self.global_step += 1

        if self.global_step <= self.warmup_steps:
            # Linear warmup
            warmup_lr = self.initial_lr * (self.global_step / self.warmup_steps)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = warmup_lr
        else:
            # Check if loss increased
            if loss > self.best_loss:
                for param_group in self.optimizer.param_groups:
                    new_lr = max(param_group["lr"] / self.decay_factor, self.min_lr)
                    param_group["lr"] = new_lr
            self.best_loss = min(self.best_loss, loss)

    def get_lr(self):
        """Return current learning rates."""
        return [param_group["lr"] for param_group in self.optimizer.param_groups]


def validate(model, device, criterion, val_loader):
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for data, label in val_loader:
            data = data.to(device)
            label = label.to(device)
            output = model(data)

            batch_loss = criterion(output, label)
            val_loss += batch_loss.item()

        return val_loss / len(val_loader)


def train(
    model: torch.nn.Module,
    model_path: str,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    epochs: int,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    learning_rate: int,
    gradient_step: int = 1,
    min_lr: float = 1e-8,
    warmup_steps: int = 1,
    lr_decay_factor: int = 2,
    patience: int = 0,
):
    """Train the model and return the best check point.

    Batch accumulation, Early stopping, Warmup scheduler,
    and Learning rate scheduler are included.

    :param model: Model to train.
    :param model_path: Path to save the best model.
    :param device: Torch device (cpu or cuda).
    :param optimizer: Optimizer for training.
    :param criterion: Loss function.
    :param epochs: Maximum number of epochs.
    :param train_loader: Training data loader.
    :param val_loader: Validation data loader.
    :param learning_rate: Learning rate.
    :param gradient_step: Set gradient_step=1 to disable gradient accumulation.
    :param min_lr: Minimum learning rate (default: 1e-8).
    :param warmup_steps: Set warmup_steps=1 to disable warmup (default).
    :param lr_decay_factor: Learning rate decay factor (default: 2).
    :param patience: Number of epochs to wait before early stopping (default: 0).
    """
    epoch_trange = trange(1, epochs + 1)
    scheduler = WarmupScheduler(
        optimizer, learning_rate, min_lr, warmup_steps, lr_decay_factor
    )
    early_stopper = EarlyStopping(patience, model_path)

    model.zero_grad()

    for epoch_id in epoch_trange:
        model.train()
        train_loss = 0
        for batch_id, (data, label) in enumerate(train_loader, start=1):

            data = data.to(device)
            label = label.to(device)
            output = model(data)

            batch_loss = criterion(output, label)
            train_loss += batch_loss.item()

            batch_loss /= gradient_step
            batch_loss.backward()

            # Gradient Accumulation
            if batch_id % gradient_step == 0:
                optimizer.step()
                model.zero_grad()

        # Validate Training Epoch
        train_loss /= len(train_loader)
        val_loss = validate(model, device, criterion, val_loader)
        tqdm.write(
            f"Epoch {epoch_id}, Train-Loss: {train_loss:.5f},  Val-Loss: {val_loss:.5f}"
        )

        # Early stopping
        if early_stopper.should_stop(val_loss, model, epoch_id):
            break

        # Learning Rate Scheduling
        scheduler.step(val_loss)

    check_point = early_stopper.check_point
    tqdm.write(f"\n--Check point: [Epoch: {check_point}]")

    return check_point
