import argparse

import torch
from torch.utils.data import DataLoader

from utils import device, evaluate, EEGDataset
from models.transformer import ViTransformer

parser = argparse.ArgumentParser(description="EEG-ViTransformer")
parser.add_argument("--dataset", help="EEG dataset path")
parser.add_argument("--fp16", default=False, action=argparse.BooleanOptionalAction)
args = parser.parse_args()


device = device(force_cuda=False)

# Dataset
dataset = EEGDataset(args.dataset)
dataloader = DataLoader(dataset, batch_size=4)

# Load pre-trained model
TRAINED_VIT_PATH = "./log/ieee-transformer_250303001232982598_3.pt"
TRAINED_VIT_CONFIG = {
    "input_channel": 19,
    "seq_length": 9250,
    "embed_dim": 64,
    "num_heads": 4,
    "num_blocks": 4,
    "block_hidden_dim": 128,
    "fc_hidden_dim": 32,
    "num_classes": 2,
}
model = ViTransformer(**TRAINED_VIT_CONFIG)
model.load_state_dict(
    torch.load(TRAINED_VIT_PATH, map_location=device, weights_only=True)
)
model.eval()

# Evaluate
metrics = evaluate(model, device, dataloader, enable_fp16=args.fp16)
print(metrics)
