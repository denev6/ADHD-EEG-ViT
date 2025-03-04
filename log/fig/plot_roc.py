import torch
from torch.utils.data import DataLoader
from utils import IEEEDataConfig, device, EEGDataset, plot_roc
from models.transformer import ViTransformer, TransformerConfig

device = device()

model_config = TransformerConfig(
    embed_dim=64,
    num_heads=4,
    num_blocks=4,
    block_hidden_dim=64,
    fc_hidden_dim=32,
    dropout=0.0,
)
data_config = IEEEDataConfig()
model = ViTransformer(
    input_channel=data_config.channels,
    seq_length=data_config.length,
    embed_dim=model_config.embed_dim,
    num_heads=model_config.num_heads,
    num_blocks=model_config.num_blocks,
    block_hidden_dim=model_config.block_hidden_dim,
    fc_hidden_dim=model_config.fc_hidden_dim,
    num_classes=data_config.num_classes,
    dropout_p=model_config.dropout,
).to(device)

model_path = "/.../model.pt"
trained_weights = torch.load(model_path, weights_only=True, map_location=device)
model.load_state_dict(trained_weights)

test_data_path = "/.../test.pt"
test_dataset = EEGDataset(test_data_path)
test_dataloader = DataLoader(test_dataset, batch_size=4)

plot_roc(model, device, test_dataloader, enable_fp16=True)
