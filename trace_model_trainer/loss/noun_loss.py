from copy import deepcopy
from typing import Dict, Iterable, List

import torch
from sentence_transformers.SentenceTransformer import SentenceTransformer
from torch import Tensor, nn


class NounLoss(nn.Module):
    def __init__(self, model: SentenceTransformer, important_words: List[str]):
        super(NounLoss, self).__init__()
        self.og_model = deepcopy(model)
        self.model = model
        self.important_words = important_words
        self.idx2word = {v: k for k, v in self.model.tokenizer.vocab.items()}

    def forward(self, sentence_features: Iterable[Dict[str, torch.Tensor]], labels: torch.Tensor):
        # Initialize total loss
        total_loss = 0.0
        mse_loss = nn.MSELoss()

        for i, sentence_feature in enumerate(sentence_features):
            # Forward pass through the model to get embeddings
            output = self.model(sentence_feature)

            # Original sentence embedding for similarity loss
            sentence_embedding = output["sentence_embedding"]
            # og_embedding = self.og_model(sentence_feature)["sentence_embedding"]
            token_contributions = extract_token_contributions(output)

            # Convert input_ids to words and identify important and unimportant words
            input_ids = sentence_feature["input_ids"]

            word_loss = 0
            for idx, input_id_list in enumerate(input_ids):
                words = [self.idx2word[token_id.item()] for token_id in input_id_list]

                # Update masks based on important words
                for j, word in enumerate(words):
                    word_loss += token_contributions[idx, j]

            # Compute similarity loss to retain original sentence embedding quality
            # similarity_loss = mse_loss(sentence_embedding, og_embedding)

            # Combine all losses
            a = 1
            # b = 1
            total_loss += (a * word_loss)  # + (similarity_loss * b)

        # Return the average loss over the mini-batch
        return total_loss / len(sentence_features)

    def get_config_dict(self):
        return {"important_words": self.important_words}

    @property
    def citation(self) -> str:
        return """"""


def extract_token_contributions(output: Dict[str, Tensor]):
    attention_mask = output["attention_mask"]
    token_embeddings = output["token_embeddings"]  # Shape: (batch_size, seq_len, hidden_dim)
    sentence_embedding = output["sentence_embedding"]  # Shape: (batch_size, hidden_dim)

    # Compute dot product scores between each token embedding and the sentence embedding
    dot_product_scores = torch.sum(token_embeddings * sentence_embedding.unsqueeze(1), dim=2)
    dot_product_scores = dot_product_scores * attention_mask
    # Normalize the scores to get token contributions
    token_contributions = dot_product_scores / (torch.sum(dot_product_scores, dim=1, keepdim=True) + 1e-9)

    return token_contributions
