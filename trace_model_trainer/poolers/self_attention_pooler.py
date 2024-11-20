from typing import Dict

import torch
from torch import Tensor, nn, softmax


class SelfAttentionBlock(nn.Module):
    def __init__(self, embedding_dim):
        super(SelfAttentionBlock, self).__init__()
        self.embedding_dim = embedding_dim
        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)

        # Aggregation FFN now outputs a single vector per sequence
        self.aggregation = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.ReLU(),
            nn.Linear(embedding_dim * 4, embedding_dim),
            nn.Tanh()  # Optional non-linearity for compact embedding
        )

    def forward(self, features: Dict[str, Tensor]):
        token_embeddings = features["token_embeddings"]  # (batch_size, seq_len, embedding_dim)
        attention_mask = features["attention_mask"]  # (batch_size, seq_len)

        # Linear projections for query, key, value
        queries = self.query(token_embeddings)  # (batch_size, seq_len, embedding_dim)
        keys = self.key(token_embeddings)  # (batch_size, seq_len, embedding_dim)
        values = self.value(token_embeddings)  # (batch_size, seq_len, embedding_dim)

        # Compute scaled attention scores
        attention_scores = torch.bmm(queries, keys.transpose(1, 2))  # (batch_size, seq_len, seq_len)
        attention_scores /= (self.embedding_dim ** 0.5)  # Scaling

        # Apply mask: large negative values for padded tokens
        attention_scores = attention_scores.masked_fill(attention_mask.unsqueeze(1) == 0, float('-inf'))

        # Normalize scores
        attention_weights = softmax(attention_scores, dim=-1)  # (batch_size, seq_len, seq_len)

        # Weighted sum of values
        attention_matrix = torch.bmm(attention_weights, values)  # (batch_size, seq_len, embedding_dim)

        # Aggregate using the feedforward network
        # Instead of mean pooling, aggregate via learned feedforward network
        # Flatten sequence dimension and apply aggregation FFN
        aggregated_embedding = self.aggregation(attention_matrix.mean(dim=1))  # (batch_size, embedding_dim)

        return {"sentence_embedding": aggregated_embedding}


class SelfAttentionPooler(nn.Module):
    def __init__(self, embedding_dim, n_heads: int = 3):
        super(SelfAttentionPooler, self).__init__()
        self.embedding_dim = embedding_dim
        self.heads = nn.ModuleList([SelfAttentionBlock(self.embedding_dim) for _ in range(n_heads)])

    def forward(self, features: Dict[str, Tensor]):
        # Ensure all heads are applied properly
        outputs = [head(features)["sentence_embedding"] for head in self.heads]
        embeddings = torch.stack(outputs, dim=0).sum(dim=0)  # Combine head outputs
        return {"sentence_embedding": embeddings}

    def save(self, output_path: str):
        """Saves weights of matrix weights"""
        state_dict = {"embedding_dim": self.embedding_dim, "n_heads": len(self.heads)}
        for i, head in enumerate(self.heads):
            state_dict[f"head_{i}"] = head.state_dict()
        torch.save(state_dict, f"{output_path}/self_attention_pooler.pt")

    @classmethod
    def load(cls, input_path: str):
        state_dict = torch.load(f"{input_path}/self_attention_pooler.pth")

        embedding_dim = state_dict.pop("embedding_dim")
        n_heads = state_dict.pop("n_heads")

        pooling_layer = cls(embedding_dim=embedding_dim, n_heads=n_heads)
        for i in range(n_heads):
            pooling_layer.heads[i].load_state_dict(state_dict[f"head_{i}"])
        return pooling_layer
