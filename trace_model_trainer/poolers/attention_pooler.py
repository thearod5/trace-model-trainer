from typing import Dict

import torch
import torch.nn as nn
from torch import Tensor


class AttentionPooler(nn.Module):
    def __init__(self, hidden_size, num_attention_heads=8, num_transformer_layers=1):
        super(AttentionPooler, self).__init__()

        # Define a Transformer Encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_attention_heads
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_transformer_layers
        )

        # Define the Multihead Attention layer
        self.multihead_attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_attention_heads)

        # Define the MLP
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, features: Dict[str, Tensor]):
        token_embeddings = features["token_embeddings"]
        attention_mask = features["attention_mask"]

        # Transpose the token_embeddings to match the input shape required by MultiheadAttention
        # MultiheadAttention expects (seq_length, batch_size, hidden_size)
        token_embeddings = token_embeddings.transpose(0, 1)

        # Apply the transformer encoder
        transformer_output = self.transformer_encoder(token_embeddings, src_key_padding_mask=attention_mask == 0)

        # Apply the Multihead Attention using the transformer output as Q, K, and V
        attn_output, _ = self.multihead_attention(
            query=transformer_output,
            key=transformer_output,
            value=transformer_output,
            key_padding_mask=attention_mask == 0
        )

        # Transpose the output back to (batch_size, seq_length, hidden_size)
        attn_output = attn_output.transpose(0, 1)

        # Take the mean of the attention output across the sequence length to get a single vector
        sentence_embeddings = attn_output.mean(dim=1)

        # Pass through the MLP
        sentence_embeddings = self.mlp(sentence_embeddings)

        return {
            "sentence_embedding": sentence_embeddings
        }

    def save(self, output_path: str):
        # Save the state of the transformer encoder, multi-head attention, and MLP
        torch.save({
            'transformer_encoder': self.transformer_encoder.state_dict(),
            'multihead_attention': self.multihead_attention.state_dict(),
            'mlp': self.mlp.state_dict()
        }, f"{output_path}/custom_attention_pooling.pth")

    @classmethod
    def load(cls, input_path: str):
        # Set your model parameters
        hidden_size = 384  # Set this to your model's hidden size
        num_attention_heads = 8  # Set this to your model's number of attention heads
        num_transformer_layers = 1  # Set this to the number of transformer layers

        # Initialize the pooling layer
        pooling_layer = cls(hidden_size, num_attention_heads, num_transformer_layers)

        # Load the saved state
        state_dict = torch.load(f"{input_path}/custom_attention_pooling.pth")
        pooling_layer.transformer_encoder.load_state_dict(state_dict['transformer_encoder'])
        pooling_layer.multihead_attention.load_state_dict(state_dict['multihead_attention'])
        pooling_layer.mlp.load_state_dict(state_dict['mlp'])

        return pooling_layer
