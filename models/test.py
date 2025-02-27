import unittest
import torch

from transformer import AttentionBlock, Transformer, ViTransformer


class TestTransformer(unittest.TestCase):
    def test_attention_block_output_shape(self):
        batch = 4
        embed_dim = 64

        block = AttentionBlock(embed_dim=embed_dim, num_heads=1, hidden_dim=1)
        input = torch.empty(batch, 1, embed_dim)
        output = block(input)
        self.assertEqual(
            input.shape, output.shape, "Attention block output shape mismatch"
        )

    def test_transformer_output_shape(self):
        batch = 4
        input_channel = 3
        seq_length = 4
        num_classes = 2

        model = Transformer(
            input_channel=input_channel,
            seq_length=seq_length,
            num_heads=1,
            num_blocks=1,
            block_hidden_dim=1,
            fc_hidden_dim=1,
            num_classes=2,
        )
        input = torch.empty(batch, seq_length, input_channel)
        output = model(input)
        self.assertEqual(
            torch.Size([batch, num_classes]),
            output.shape,
            "Transformer output shape mismatch",
        )

    def test_vitransformer_output_shape(self):
        batch = 4
        input_channel = 3
        seq_length = 4
        num_classes = 2

        model = ViTransformer(
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
            torch.Size([batch, num_classes]),
            output.shape,
            "viTransformer output shape mismatch",
        )
