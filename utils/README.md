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
