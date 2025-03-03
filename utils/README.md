# Utilities for Model Training

Available classes and functions: [list](__init__.py)

## Google Colab

Make sure to run this code before attempting to import the module.

```python
# Mount Google Drive for Colab env
import sys
from google.colab import drive

drive.mount("/content/drive", force_remount=False)
sys.path.append("/content/drive/MyDrive")
```

### FP16 precision

Colab's GPUs support FP16 precision, which significantly sped up training by roughly threefold. (Specifically, 5 epochs took 207.32 seconds with standard precision to train the transformer and only 61.92 seconds with FP16.)

## Train

```python
import torch
from utils import train, device

model = CNN(...)
optimizer = torch.optim.SGD(model.parameters(), ...)
criterion = torch.nn.CrossEntropyLoss(...)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, ...)

epochs = 10
model_path = "dummy_model.pt"

train_dataloader = torch.utils.data.DataLoader(...)
val_dataloader = torch.utils.data.DataLoader(...)

check_point = train(
    model=model,
    device=device(),
    model_path=model_path,
    optimizer=optimizer,
    criterion=criterion,
    epochs=epochs,
    train_loader=train_dataloader,
    val_loader=val_dataloader,
    scheduler=scheduler,
    enable_fp16=True,
)
```

### Cross Validation

```python
import torch
from utils import train_with_kfold, device

model = CNN # without init
optimizer = torch.optim.SGD # without init
criterion = torch.nn.CrossEntropyLoss(...)

model_path = "dummy_model.pt"
epochs = 10
k_folds = 5
batch_size = 4

dataset = torch.utils.data.TensorDataset(...) # Dataset, not Dataloader

check_point, best_model = train_with_kfold(
    k_folds=k_folds,
    device=device(),
    model_path=model_path,
    model_class=model,
    model_params={"hidden_dim": 64, "dropout": 0.1},
    optimizer_class=optimizer,
    optimizer_params={"lr": 1e-3},
    criterion=criterion,
    epochs=epochs,
    train_dataset=dataset,
    batch=batch_size,
)
```