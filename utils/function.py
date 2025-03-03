import os
import json
import gc
import warnings
from dataclasses import is_dataclass, asdict
import torch
import torch.nn.functional as F
from sympy.stats.rv import probability
from torch.amp import autocast
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    roc_curve,
    roc_auc_score,
)
import matplotlib.pyplot as plt


# Google Drive for Colab env
def join_drive_path(*args):
    """Join Google Drive path"""
    return os.path.join("/content/drive/MyDrive", *args)


# Torch
def clear_cache():
    gc.collect()
    torch.cuda.empty_cache()


def device(force_cuda=True) -> torch.device:
    has_cuda = torch.cuda.is_available()
    if force_cuda:
        assert has_cuda, "CUDA is not available."
        return torch.device("cuda")
    return torch.device("cuda") if has_cuda else torch.device("cpu")


def inference(
    model: torch.nn.Module,
    device: torch.device,
    data_loader: torch.utils.data.DataLoader,
    enable_fp16: bool = False,
):
    """Return the output of the model

    :returns y_pred, y_true: array of predicted labels and true labels
    """
    # FP16 precision
    if enable_fp16:
        assert torch.amp.autocast_mode.is_autocast_available(
            str(device)
        ), "Unable to use autocast on current device."

    with torch.no_grad():
        with autocast(
            device_type=str(device), enabled=enable_fp16, dtype=torch.float16
        ):
            model.to(device)
            model.eval()
            y_pred = list()
            y_true = list()
            for data, label in data_loader:
                data = data.to(device)
                output = model(data)
                probs = F.softmax(output.float(), dim=1)
                y_pred.extend(probs.detach().cpu().numpy())
                y_true.extend(label.numpy())

            return y_pred, y_true


def evaluate(
    model: torch.nn.Module,
    device: torch.device,
    data_loader: torch.utils.data.DataLoader,
    enable_fp16: bool = False,
):
    """Return metrics for test set

    :returns metrics: { accuracy, f1-score, recall, auc }
    """
    y_pred, y_true = inference(model, device, data_loader, enable_fp16)
    y_pred = np.argmax(y_pred, axis=1)

    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    auc_value = roc_auc_score(y_true, y_pred)

    return {
        "accuracy": accuracy,
        "f1-score": f1,
        "recall": recall,
        "auc": auc_value,
    }


# Visualize
def plot_roc(
    model: torch.nn.Module,
    device: torch.device,
    data_loader: torch.utils.data.DataLoader,
    enable_fp16: bool = False,
    title: str = "ROC Curve",
):

    y_pred, y_true = inference(model, device, data_loader, enable_fp16)
    fpr, tpr, _ = roc_curve(y_true, np.array(y_pred)[:, 1])

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color="blue")
    plt.plot([0, 1], [0, 1], color="grey", linestyle="--")  # Baseline

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.grid()
    plt.show()


# Others
def ignore_warnings():
    warnings.filterwarnings("ignore")


def fix_random_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def log_json(file_path: str, *args, **kwargs):
    """Save logs to a JSON file.

    :param file_path: path to save the logs
    :param kwargs: {name: dataclass} or {name: value} pairs to save
    """
    _, ext = os.path.splitext(file_path)
    assert ext == ".json", "File path must be a JSON file."

    logs = dict()

    # arguments
    for arg in args:
        if is_dataclass(arg):
            arg = asdict(arg)
        for name, attr in arg.items():
            _safe_update_dict(logs, str(name).strip(), attr)

    # keyword arguments
    for name, value in kwargs.items():
        if is_dataclass(value):
            # Do not use 'asdict' since 'Config' allows extra values
            # which are not included in dataclasses.fields.
            value = value.__dict__
        _safe_update_dict(logs, str(name).strip(), value)

    with open(file_path, "w") as f:
        json.dump(logs, f, indent=2)

    return logs


def _safe_update_dict(d, k, v):
    if k in d:
        raise KeyError(f"Duplicate key: {k}")
    if v is not None:
        d[k] = v
