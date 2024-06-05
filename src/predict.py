from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from tdata.trace_dataset import TraceDataset
from tdata.types import TracePrediction
from utils import t_id_creator


def predict_scores(model: SentenceTransformer, dataset: TraceDataset) -> List[TracePrediction]:
    content_ids = list(dataset.artifact_map.keys())
    content_set = list(dataset.artifact_map.values())
    embeddings = model.encode(content_set, convert_to_tensor=False, show_progress_bar=True)
    embedding_map = {a_id: e for a_id, e in zip(content_ids, embeddings)}

    for source_artifact_ids, target_artifact_ids in dataset.get_layer_iterator():
        source_embeddings = np.stack([embedding_map[s_id] for s_id in source_artifact_ids])
        target_embeddings = np.stack([embedding_map[t_id] for t_id in target_artifact_ids])

        scores = cosine_similarity(source_embeddings, target_embeddings)
        predictions = []
        for i, s_id in enumerate(source_artifact_ids):
            for j, t_id in enumerate(target_artifact_ids):
                score = scores[i, j]
                t_id = t_id_creator(source=s_id, target=t_id)
                label = dataset.trace_map[t_id]['label'] if t_id in dataset.trace_map else 0
                predictions.append(
                    TracePrediction(
                        source=s_id,
                        target=t_id,
                        label=label,
                        score=score
                    )
                )

    return predictions
