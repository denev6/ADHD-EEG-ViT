import unittest
import torch

from transformer import AttentionBlock, Transformer


class TestTransformer(unittest.TestCase):
    def test_attention_block_output_shape(self):
        batch = 4
        embed_dim = 64

        block = AttentionBlock(embed_dim=embed_dim, num_heads=1, hidden_dim=1)
        input = torch.empty(batch, 1, embed_dim)
        output = block(input)
        self.assertEqual(
            output.shape, input.shape, "Attention block output shape mismatch"
        )

    def test_transformer_output_shape(self):
        batch = 2
        input_channel = 3
        seq_length = 4
        num_classes = 2

        model = Transformer(
            input_channel=input_channel,
            seq_length=seq_length,
            embed_dim=1,
            num_heads=1,
            num_blocks=1,
            block_hidden_dim=1,
            fc_hidden_dim=1,
            num_classes=2,
        )
        input = torch.empty(batch, input_channel, seq_length)
        output = model(input)
        self.assertEqual(
            output.shape,
            torch.Size([batch, num_classes]),
            "Transformer output shape mismatch",
        )
