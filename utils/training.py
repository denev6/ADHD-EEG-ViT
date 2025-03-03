import torch
from torch import autocast
from torch.amp import GradScaler
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import KFold
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

    @property
    def best_loss(self):
        return self._min_loss


class WarmupScheduler(object):
    """Warmup learning rate and dynamically adjusts learning rate based on validation loss.

    When the loss increases, the learning rate will be divided by decay_factor.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        lr: float,
        min_lr: float = 1e-6,
        warmup_steps: int = 10,
        decay_factor: float = 0.1,
    ):
        """Initialize Warmup Scheduler.

        :param optimizer: Optimizer for training.
        :param lr: Learning rate.
        :param min_lr: Minimum learning rate.
        :param warmup_steps: Number of warmup steps.
        :param decay_factor: Factor to multiply learning rate when loss increases.
        """
        self.optimizer = optimizer
        self.initial_lr = lr
        self.min_lr = min_lr
        self.decay_factor = decay_factor

        # If user set warmup_steps=0, then set warmup_steps=1 to avoid ZeroDivisionError.
        self.warmup_steps = max(warmup_steps, 1)

        assert (
            self.initial_lr >= self.min_lr
        ), f"Learning rate must be greater than min_lr({self.min_lr})"
        assert 0 < self.decay_factor < 1, "Decay factor must be less than 1.0."

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
                    new_lr = max(param_group["lr"] * self.decay_factor, self.min_lr)
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


def _train(
    model: torch.nn.Module,
    device: torch.device,
    model_path: str,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    epochs: int,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    gradient_step: int = 1,
    patience: int = 0,
    enable_fp16: bool = False,
    scheduler=None,
):
    if scheduler is not None:
        assert callable(
            getattr(scheduler, "step", None)
        ), "Scheduler must have a step() method."

    epoch_trange = trange(1, epochs + 1)
    early_stopper = EarlyStopping(patience, model_path)
    scaler = GradScaler(device=str(device), enabled=enable_fp16)

    model.to(device)
    criterion.to(device)

    model.zero_grad()

    for epoch_idx in epoch_trange:
        model.train()
        train_loss = 0
        for batch_id, (data, label) in enumerate(train_loader, start=1):
            data = data.to(device)
            label = label.to(device)

            with autocast(
                device_type=str(device), enabled=enable_fp16, dtype=torch.float16
            ):
                output = model(data)
                batch_loss = criterion(output, label)

            train_loss += batch_loss.item()

            # Scale loss to prevent under/overflow
            scaler.scale(batch_loss / gradient_step).backward()

            # Gradient Accumulation
            if batch_id % gradient_step == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        # Validate Training Epoch
        train_loss /= len(train_loader)
        val_loss = validate(model, torch.device("cuda"), criterion, val_loader)
        tqdm.write(
            f"Epoch {epoch_idx}, Train-Loss: {train_loss:.5f},  Val-Loss: {val_loss:.5f}"
        )

        # Early stopping
        if early_stopper.should_stop(val_loss, model, epoch_idx):
            break

        # Learning Rate Scheduling
        if scheduler is not None:
            if isinstance(scheduler, WarmupScheduler):
                scheduler.step(val_loss)
            else:
                scheduler.step(epoch_idx)

    return early_stopper.check_point, early_stopper.best_loss


def train(
    model: torch.nn.Module,
    device: torch.device,
    model_path: str,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    epochs: int,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    gradient_step: int = 1,
    patience: int = 0,
    enable_fp16: bool = False,
    scheduler=None,
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
    :param gradient_step: Set gradient_step=1 to disable gradient accumulation.
    :param patience: Number of epochs to wait before early stopping (default: 0).
    :param enable_fp16: Enable FP16 precision training (default: False).
    :param scheduler: Learning rate scheduler (default: None).
    """
    if enable_fp16:
        assert torch.amp.autocast_mode.is_autocast_available(
            str(device)
        ), "Unable to use autocast on current device."

    check_point, _ = _train(
        model,
        device,
        model_path,
        optimizer,
        criterion,
        epochs,
        train_loader,
        val_loader,
        gradient_step,
        patience,
        enable_fp16,
        scheduler,
    )
    return check_point


def train_with_kfold(
    k_folds: int,
    model_class: torch.nn,
    device: torch.device,
    model_path: str,
    optimizer_class: torch.optim,
    criterion: torch.nn.Module,
    epochs: int,
    train_dataset: torch.utils.data.Dataset,
    batch: int,
    model_params: dict = None,
    optimizer_params: dict = None,
    gradient_step: int = 1,
    patience: int = 0,
    enable_fp16: bool = False,
    scheduler_class=None,
    scheduler_params: dict = None,
):
    """Train the model and return the best check point.

    Batch accumulation, Early stopping, Warmup scheduler,
    and Learning rate scheduler are included.

    :param k_folds: Number of folds for K-fold cross validation.
    :param model_class: Model class to train.
    :param model_params: Parameters for 'model_class'.
    :param model_path: Path to save the best model.
    :param device: Torch device (cpu or cuda).
    :param optimizer_class: Optimizer class for training.
    :param optimizer_params: Parameters for 'optimizer_class'. model.parameters will be called automatically.
    :param criterion: Loss function.
    :param epochs: Maximum number of epochs.
    :param train_dataset: Training data set.
    :param batch: Batch size for training.
    :param gradient_step: Set gradient_step=1 to disable gradient accumulation.
    :param patience: Number of epochs to wait before early stopping (default: 0).
    :param enable_fp16: Enable FP16 precision training (default: False).
    :param scheduler_class: Learning rate scheduler class. optimizer will be called automatically. (default: None).
    :param scheduler_params: Parameters for 'scheduler_class'.  (default: None).
    """
    if enable_fp16:
        assert torch.amp.autocast_mode.is_autocast_available(
            str(device)
        ), "Unable to use autocast on current device."

    kf = KFold(n_splits=k_folds, shuffle=True)
    best_fold = 0
    best_check_point = 0
    best_val_loss = float("inf")
    model_name, ext = model_path.rsplit(".", 1)

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_dataset), start=1):
        torch.cuda.empty_cache()
        print(f"\n===== Fold {fold} =====")

        train_dataset_fold = Subset(train_dataset, train_idx)
        val_dataset_fold = Subset(train_dataset, val_idx)
        train_loader = DataLoader(train_dataset_fold, batch_size=batch, shuffle=True)
        val_loader = DataLoader(val_dataset_fold, batch_size=batch, shuffle=False)
        model_path = f"{model_name}_{fold}.{ext}"

        if model_params is None:
            model_params = {}
        if optimizer_params is None:
            optimizer_params = {}

        model = model_class(**model_params)
        optimizer = optimizer_class(model.parameters(), **optimizer_params)

        if scheduler_class is not None:
            if scheduler_params is None:
                scheduler_params = {}
            scheduler = scheduler_class(optimizer, **scheduler_params)
        else:
            scheduler = None

        check_point, val_loss = _train(
            model,
            device,
            model_path,
            optimizer,
            criterion,
            epochs,
            train_loader,
            val_loader,
            gradient_step,
            patience,
            enable_fp16,
            scheduler,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_check_point = check_point
            best_fold = fold

    best_model_path = f"{model_name}_{best_fold}.{ext}"
    return best_check_point, best_model_path
