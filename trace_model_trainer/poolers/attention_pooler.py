from typing import Dict

import torch
import torch.nn as nn
from torch import Tensor


class AttentionPooler(nn.Module):
    def __init__(self, hidden_size, num_attention_heads=8):
        super(AttentionPooler, self).__init__()

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

        # Create an attention mask in the format expected by MultiheadAttention
        # It should be of shape (batch_size, seq_length) and contain True for positions to be masked
        # Convert attention_mask from (batch_size, seq_length) to (seq_length, batch_size)
        attention_mask = attention_mask == 0  # Convert to a boolean mask where True indicates padding
        # attention_mask = attention_mask.transpose(0, 1)

        # Apply the Multihead Attention
        attn_output, _ = self.multihead_attention(token_embeddings,
                                                  token_embeddings,
                                                  token_embeddings,
                                                  key_padding_mask=attention_mask)

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
        # Save the pooling layer configuration
        torch.save(self.state_dict(), f"{output_path}/custom_attention_pooling.pth")

    @classmethod
    def load(cls, input_path: str):
        # Load the pooling layer configuration
        hidden_size = 384  # Set this to your model's hidden size
        num_attention_heads = 8  # Set this to your model's number of attention heads
        pooling_layer = cls(hidden_size, num_attention_heads)
        pooling_layer.load_state_dict(torch.load(f"{input_path}/custom_attention_pooling.pth"))
        return pooling_layer
