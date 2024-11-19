from typing import Dict

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class WeightedMeanPooler(nn.Module):
    def __init__(self, embedding_dim):
        super(WeightedMeanPooler, self).__init__()
        self.embedding_dim = embedding_dim

        # A simple feed-forward network to compute weights for each token
        self.weight_layer = nn.Linear(embedding_dim, 1)

    def forward(self, features: Dict[str, Tensor]):
        token_embeddings = features["token_embeddings"]  # (batch_size, sequence_length, embedding_dim)
        attention_mask = features["attention_mask"]  # (batch_size, sequence_length)

        # Compute raw weights for each token
        raw_weights = self.weight_layer(token_embeddings).squeeze(-1)  # (batch_size, sequence_length)

        # Mask the raw weights to ignore padding tokens
        raw_weights = raw_weights.masked_fill(attention_mask == 0, float('-inf'))

        # Apply softmax to get normalized weights
        normalized_weights = F.softmax(raw_weights, dim=-1)  # (batch_size, sequence_length)

        # Compute the weighted mean of the token embeddings
        weighted_embeddings = token_embeddings * normalized_weights.unsqueeze(-1)  # (batch_size, sequence_length, embedding_dim)
        sentence_embedding = weighted_embeddings.sum(dim=1)  # (batch_size, embedding_dim)

        return {
            "sentence_embedding": sentence_embedding,
            "token_weights": normalized_weights
        }

    def save(self, output_path: str):
        torch.save({
            "embedding_dim": self.embedding_dim,
            "weight_layer": self.weight_layer.state_dict()
        }, f"{output_path}/weight_layer.pt")

    @classmethod
    def load(cls, input_path: str):
        state_dict = torch.load(f"{input_path}/weight_layer.pth")
        pooling_layer = cls(state_dict["embedding_dim"])
        pooling_layer.weight_layer.load_state_dict(state_dict['weight_layer'])
        return pooling_layer
