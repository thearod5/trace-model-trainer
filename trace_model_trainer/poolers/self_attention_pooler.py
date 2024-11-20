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

    def forward(self, features: Dict[str, Tensor]):
        token_embeddings = features["token_embeddings"]
        attention_mask = features["attention_mask"]

        x = token_embeddings

        # x: (batch_size, sequence_length, embedding_dim)
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)

        # Compute attention scores and apply scaling
        attention_scores = torch.bmm(queries, keys.transpose(1, 2)) / (self.embedding_dim ** 0.5)

        # Apply the attention mask to ignore padding tokens
        # Masked positions are set to a large negative value to ensure they have no influence
        attention_scores = attention_scores.masked_fill(attention_mask.unsqueeze(1) == 0, float('-inf'))

        # Normalize the attention scores using softmax
        attention_weights = softmax(attention_scores, dim=-1)

        # Compute the weighted sum of values
        weighted_sum = torch.bmm(attention_weights, values).sum(dim=1)  # Average across the sequence dimension

        return {"sentence_embedding": weighted_sum}

    def save(self, output_path: str):
        """Saves weights of matrix weights"""
        torch.save({
            "embedding_dim": self.embedding_dim,
            "query": self.query.state_dict(),
            "key": self.key.state_dict(),
            "value": self.value.state_dict()
        }, f"{output_path}/self_attention_pooler.pt")

    @classmethod
    def load(cls, input_path: str):
        state_dict = torch.load(f"{input_path}/self_attention_pooler.pth")

        pooling_layer = cls(state_dict["embedding_dim"])
        pooling_layer.query.load_state_dict(state_dict['query'])
        pooling_layer.key.load_state_dict(state_dict['key'])
        pooling_layer.value.load_state_dict(state_dict['value'])

        return pooling_layer


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
