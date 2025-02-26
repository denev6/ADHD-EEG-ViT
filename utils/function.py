import os
import json
import gc
import warnings
from dataclasses import is_dataclass, asdict
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_curve, auc
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


def evaluate(
    model: torch.nn.Module,
    device: torch.device,
    data_loader: torch.utils.data.DataLoader,
):
    """Return metrics for test set

    :returns metrics: { accuracy, f1-score, recall, auc, roc-curve(fpr, tpr) }
    """
    model.eval()
    y_pred = list()
    y_true = list()

    with torch.no_grad():
        for data, label in data_loader:
            data = data.to(device)
            output = model(data)

            y_pred.extend(output.argmax(1).detach().cpu().numpy())
            y_true.extend(label.numpy())

        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        auc_value = auc(fpr, tpr)

        return {
            "accuracy": accuracy,
            "f1-score": f1,
            "recall": recall,
            "auc": auc_value,
            "roc-curve": (fpr, tpr),
        }


# Visualize
def plot_roc(fpr, tpr, title: str = "ROC Curve"):
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


def log_json(file_path: str, *configs, **kwargs):
    """Save logs to a JSON file.

    :param file_path: path to save the logs
    :param configs: dataclass objects to save
    :param kwargs: additional key-value pairs to save
    """
    _, ext = os.path.splitext(file_path)
    assert ext == ".json", "File path must be a JSON file."

    logs = dict()

    for config in configs:
        if not is_dataclass(config):
            raise ValueError("Config must be a dataclass object.")
        for key, value in asdict(config).items():
            _safe_update_dict(logs, key, value)

    for key, value in kwargs.items():
        _safe_update_dict(logs, key, value)

    with open(file_path, "w") as f:
        json.dump(logs, f, indent=2)

    return logs


def _safe_update_dict(d, k, v):
    if k in d:
        raise ValueError(f"Duplicate key: {k}")
    d[k] = v
