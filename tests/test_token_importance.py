import os
from typing import Dict
from unittest import TestCase

import matplotlib.pyplot as plt
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Pooling
from sklearn.metrics.pairwise import cosine_similarity
from torch import Tensor

from trace_model_trainer.tdata.loader import load_traceability_dataset


class CustomPooling(Pooling):

    def forward(self, features: Dict[str, Tensor]):
        token_embeddings = features["token_embeddings"]
        attention_mask = features["attention_mask"]
        if not self.include_prompt and "prompt_length" in features:
            attention_mask[:, : features["prompt_length"]] = 0

        ## Pooling strategy
        output_vectors = []
        token_contributions = None

        if self.pooling_mode_mean_tokens or self.pooling_mode_mean_sqrt_len_tokens:
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(token_embeddings.size()).to(token_embeddings.dtype)
            )
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)

            # If tokens are weighted (by WordWeights layer), feature 'token_weights_sum' will be present
            if "token_weights_sum" in features:
                sum_mask = features["token_weights_sum"].unsqueeze(-1).expand(sum_embeddings.size())
            else:
                sum_mask = input_mask_expanded.sum(1)

            sum_mask = torch.clamp(sum_mask, min=1e-9)
            USE_COSINE_SIMILARITY = True
            if self.pooling_mode_mean_tokens:
                mean_embeddings = sum_embeddings / sum_mask
                output_vectors.append(mean_embeddings)

                if USE_COSINE_SIMILARITY:
                    # Calculate contribution scores using cosine similarity
                    # Normalize the token embeddings and mean embedding
                    token_embeddings_norm = torch.nn.functional.normalize(token_embeddings, p=2, dim=2)
                    mean_embedding_norm = torch.nn.functional.normalize(mean_embeddings, p=2, dim=1)

                    # Calculate cosine similarity
                    cosine_similarity_scores = torch.sum(token_embeddings_norm * mean_embedding_norm.unsqueeze(1), dim=2)

                    # Mask out padding tokens
                    cosine_similarity_scores = cosine_similarity_scores * attention_mask

                    # Normalize the scores to get a single contribution score per token
                    token_contributions = cosine_similarity_scores / (torch.sum(cosine_similarity_scores, dim=1, keepdim=True) + 1e-9)
                else:
                    # Calculate contribution scores using dot product
                    # Token contributions are computed as the dot product between each token embedding and the mean embedding
                    dot_product_scores = torch.sum(token_embeddings * mean_embeddings.unsqueeze(1), dim=2)

                    # Mask out padding tokens
                    dot_product_scores = dot_product_scores * attention_mask

                    # Normalize the scores to get a single contribution score per token
                    token_contributions = dot_product_scores / (torch.sum(dot_product_scores, dim=1, keepdim=True) + 1e-9)

            if self.pooling_mode_mean_sqrt_len_tokens:
                output_vectors.append(sum_embeddings / torch.sqrt(sum_mask))

        # ... (other pooling strategies remain unchanged)

        output_vector = torch.cat(output_vectors, 1)
        features["sentence_embedding"] = output_vector

        if token_contributions is not None:
            features["token_contributions"] = token_contributions

        return features


class TestTokenImportance(TestCase):
    def test_token_importance(self):
        dataset = load_traceability_dataset(os.path.expanduser("~/projects/trace-model-trainer/res/test"))
        k = 3
        # Select artifact
        artifact_ids = list(dataset.artifact_map.keys())[:k]
        artifact_id = artifact_ids[1]
        artifact_sentence = dataset.artifact_map[artifact_id]

        # Load the SentenceTransformer model
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
        prefix = "What is the intent of this artifact?\n\n"

        # Load Trace
        LOAD_TRACE = False
        if LOAD_TRACE:
            artifact_sentence_transformed = artifact_sentence.replace(" with ", " ")
            related_artifact_ids = dataset.trace_df[dataset.trace_df["source"] == artifact_id]["target"].to_list()
            target_embeddings = model.encode([dataset.artifact_map[r_id] for r_id in related_artifact_ids], prompt=prefix)
            sim_matrix0 = cosine_similarity([model.encode(artifact_sentence, prompt=prefix)], target_embeddings)
            sim_matrix1 = cosine_similarity([model.encode(artifact_sentence_transformed, prompt=prefix)], target_embeddings)

            print("Original:", artifact_sentence, sim_matrix0.tolist())
            print("Transformed:", artifact_sentence_transformed, sim_matrix1.tolist())

        # Get the word embedding dimension from the model's first module (usually the transformer)
        word_embedding_dimension = model.get_sentence_embedding_dimension()

        # Replace the default pooling layer with your custom pooling layer
        custom_pooling = CustomPooling(
            word_embedding_dimension=word_embedding_dimension,
            pooling_mode="mean"  # Set your desired pooling mode here
        )
        model.pooling_layer = custom_pooling  # Modify the model to use the custom pooling layer

        # Encode sentences and get features
        artifact_sentences = []
        for artifact_id in artifact_ids:
            artifact_content = dataset.artifact_map[artifact_id]
            artifact_sentences.append(artifact_content)
            artifact_sentences.append(prefix + artifact_content)

        features = model.tokenize(artifact_sentences)
        with torch.no_grad():
            output = model.forward(features)

        # Access the token contributions
        tokenizer = model.tokenizer
        token_contributions = output.get("token_contributions", None)  # Tensor of shape (batch_size, max_sequence_length)

        # Assuming token_contributions is a tensor of shape (5, 17) for 5 sentences and 17 tokens each
        # Convert the tensor to a numpy array
        contribution_scores = token_contributions.cpu().numpy()

        # Create a vertical stack of plots
        fig, axes = plt.subplots(len(artifact_sentences), 1,
                                 figsize=(10, len(artifact_sentences) * 4))  # Adjust height based on the number of sentences

        # Plot each sentence's token contributions
        for i, (sentence, ax) in enumerate(zip(artifact_sentences, axes)):
            # Tokenize the sentence to get individual tokens
            tokens = tokenizer.tokenize(sentence)

            # Get the contribution scores for the current sentence
            scores = contribution_scores[i][:len(tokens)]  # Take only the valid tokens

            # Plot the tokens with their contribution scores
            ax.bar(range(len(tokens)), scores, color="blue", alpha=0.6)
            ax.set_xticks(range(len(tokens)))
            ax.set_xticklabels(tokens, rotation=45, ha="right")
            ax.set_title(f"Sentence: {sentence}")
            ax.set_xlabel("Tokens")
            ax.set_ylabel("Contribution Score")
            ax.set_ylim(0, max(scores) + 0.05)  # Assuming scores are normalized between 0 and 1

        plt.tight_layout()  # Adjust layout to avoid overlap
        plt.show()
