from typing import List, Tuple

import torch
from sentence_transformers import SentenceTransformer
from torch import cosine_similarity


def predict_scores(model: SentenceTransformer, prediction_payload: List[Tuple[str, str]]):
    content_set = list(set([t for p in prediction_payload for t in p]))
    embeddings = model.encode(content_set, convert_to_tensor=True, show_progress_bar=True)
    embedding_map = {c: e for c, e in zip(content_set, embeddings)}
    source_embeddings = torch.stack([embedding_map[s] for s, t in prediction_payload])
    target_embeddings = torch.stack([embedding_map[t] for s, t in prediction_payload])
    scores = cosine_similarity(source_embeddings, target_embeddings).tolist()
    return scores
