from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


def _time_stamp():
    current_time = datetime.now()
    return current_time.strftime("%y%m%d%H%M%S%f")


def _format_name(name: str, max_len: int = 30) -> str:
    name = name.strip().lower().replace(" ", "-")
    return name[:max_len]


@dataclass()
class Config:
    """Configuration for training.

    :param name: name of the experiment
    :param batch: batch size
    :param epochs: number of epochs
    :param lr: learning rate
    :param grad_step: number of gradient accumulation steps
    :param warmup_steps: number of warmup steps
    :param lr_decay_factor: learning rate decay factor
    :param weight_decay: weight decay
    :param patience: patience for early stopping
    """

    id: str = field(init=False)
    name: str
    model_path: str = field(init=False)
    batch: int
    epochs: int
    lr: float
    grad_step: int = field(default=1)
    warmup_steps: Optional[int] = field(default=None)
    lr_decay_factor: int = field(default=10)
    weight_decay: Optional[float] = field(default=1e-4)
    patience: Optional[int] = field(default=0)

    def __post_init__(self):
        self.id = _time_stamp()
        self.name = _format_name(self.name)
        self.model_path = f"{self.name}_{self.id}.pt"

        if self.batch < 1:
            raise ValueError("batch must be positive integer.")
        if self.epochs < 0:
            raise ValueError("epochs must be positive integer.")
        if self.lr < 0:
            raise ValueError("lr must be positive float.")
