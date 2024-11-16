from collections import defaultdict
from typing import Dict, Union

from datasets import tqdm
from sentence_transformers.evaluation import SentenceEvaluator
from sklearn.metrics.pairwise import cosine_similarity

from trace_model_trainer.eval.utils import calculate_prediction_metrics
from trace_model_trainer.formatters.classification_formatter import ClassificationFormatter
from trace_model_trainer.tdata.trace_dataset import TraceDataset
from trace_model_trainer.tdata.types import TracePrediction


class CustomCosineEvaluator(SentenceEvaluator):
    def __init__(self, dataset: TraceDataset):
        super().__init__()
        self.dataset = ClassificationFormatter().format(dataset)
        print("Creating Custom Cosine Evaluator")

    def __call__(
            self, model: "SentenceTransformer", output_path: str = None, epoch: int = -1, steps: int = -1
    ) -> Union[float, Dict[str, float]]:
        texts = list(set(self.dataset["sentence1"]).union(set(self.dataset["sentence2"])))
        embeddings = model.encode(texts, show_progress_bar=True)
        text2embeddings = {t: e for t, e in zip(texts, embeddings)}

        query2candidates = defaultdict(list)
        for d in self.dataset:
            query2candidates[d["sentence1"]].append(d)

        predictions = []
        for query, candidates in tqdm(query2candidates.items()):
            sources = [text2embeddings[query]]
            targets = [text2embeddings[c["sentence2"]] for c in candidates]
            scores = cosine_similarity(sources, targets)[0]
            for c, score in zip(candidates, scores):
                predictions.append(TracePrediction(
                    source=c["sentence1"],
                    target=c["sentence2"],
                    label=c["label"],
                    score=score
                ))

        metrics = calculate_prediction_metrics(predictions)

        return metrics
