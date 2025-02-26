from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass(frozen=True)
class TransformerConfig:
    """Configuration for Transformer model.

    :param embed_dim: embedding dimension
    :param num_heads: number of attention heads
    :param num_blocks: number of attention blocks
    :param block_hidden_dim: dimension of attention blocks
    :param fc_hidden_dim: dimension of feed forward layers
    :param dropout: dropout probability
    """

    embed_dim: int
    num_heads: int
    num_blocks: int
    block_hidden_dim: int
    fc_hidden_dim: int
    dropout: float


class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim):
        super(AttentionBlock, self).__init__()
        self.embed_dim = embed_dim
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, input: torch.Tensor):
        torch._assert(
            input.shape[2] == self.embed_dim,
            f"Input shape must be (batch_size, seq_len, {self.embed_dim})",
        )

        # Multi-head Attention
        x, _ = self.attention(input, input, input, need_weights=False)

        # Add & Norm
        x = self.norm1(x + input)

        # Feed Forward
        x = self.feedforward(x)

        # Add & Norm
        x = self.norm2(x + input)
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        input_channel,
        seq_length,
        embed_dim,
        num_heads,
        num_blocks,
        block_hidden_dim,
        fc_hidden_dim,
        num_classes,
        dropout_p=0.0,
    ):
        super(Transformer, self).__init__()
        self.dropout_p = dropout_p
        self.signal_channel = input_channel
        self.seq_length = seq_length

        # Embedding
        self.pos_embedding = nn.Parameter(
            torch.empty(1, seq_length, embed_dim).normal_(std=0.02)
        )

        # Attention Blocks
        self.encoder = nn.ModuleList(
            [
                AttentionBlock(embed_dim, num_heads, block_hidden_dim)
                for _ in range(num_blocks)
            ]
        )

        # Decoding layers
        self.global_max_pool = nn.Sequential(
            nn.AdaptiveMaxPool1d(1), nn.Dropout(p=self.dropout_p)
        )
        self.fc = nn.Sequential(
            nn.Flatten(1, -1),
            nn.Linear(embed_dim, fc_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_p),
            nn.Linear(fc_hidden_dim, num_classes),
        )

    def forward(self, input):
        torch._assert(
            input.shape[1:] == (self.seq_length, self.signal_channel),
            f"Expected shape of (batch, {self.seq_length}, {self.signal_channel})",
        )
        x = input + self.pos_embedding

        for layer in self.encoder:
            x = layer(x)

        x = x.permute(0, 2, 1)
        # x: (-1, embed_dim, seq_len)
        x = self.global_max_pool(x)
        x = self.fc(x)
        return x


class ViTransformer(nn.Module):
    def __init__(
        self,
        input_channel,
        seq_length,
        embed_dim,
        num_heads,
        num_blocks,
        block_hidden_dim,
        fc_hidden_dim,
        num_classes,
        dropout_p=0.0,
    ):
        super(ViTransformer, self).__init__()
        self.dropout_p = dropout_p
        self.signal_channel = input_channel
        self.seq_length = seq_length

        # Embedding
        self.proj = nn.Conv1d(self.signal_channel, embed_dim, kernel_size=3, padding=1)

        self.pos_embedding = nn.Parameter(
            torch.empty(1, seq_length, embed_dim).normal_(std=0.02)
        )
        self.encoder = nn.ModuleList(
            [
                AttentionBlock(embed_dim, num_heads, block_hidden_dim)
                for _ in range(num_blocks)
            ]
        )
        self.global_max_pool = nn.Sequential(
            nn.AdaptiveMaxPool1d(1), nn.Dropout(p=self.dropout_p)
        )
        self.fc = nn.Sequential(
            nn.Flatten(1, -1),
            nn.Linear(embed_dim, fc_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_p),
            nn.Linear(fc_hidden_dim, num_classes),
        )

    def forward(self, input):
        torch._assert(
            input.shape[1:] == (self.signal_channel, self.seq_length),
            f"Expected shape of (batch, {self.signal_channel}, {self.seq_length})",
        )
        # x: (-1, embed_dim, seq_len)
        x = self.proj(input)

        # Self-attention requires shape of (-1, seq_len, embed_dim)
        x = x.permute(0, 2, 1)
        x = x + self.pos_embedding

        for layer in self.encoder:
            x = layer(x)

        x = x.permute(0, 2, 1)
        # x: (-1, embed_dim, seq_len)
        x = self.global_max_pool(x)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    from torchinfo import summary

    # Original EEG-Transformer from the paper (Table 1)
    model = Transformer(
        input_channel=56,
        seq_length=385,
        embed_dim=56,
        num_heads=4,
        num_blocks=6,
        num_classes=3,
        block_hidden_dim=56,
        fc_hidden_dim=64,
        dropout_p=0.5,
    )
    summary(model, input_size=(1, 385, 56))
