from typing import Dict, Union

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
        embeddings = model.encode(texts)
        text2embeddings = {t: e for t, e in zip(texts, embeddings)}

        predictions = []
        for d in self.dataset:
            score = cosine_similarity([text2embeddings[d["sentence1"]]], [text2embeddings[d["sentence2"]]])[0][0]
            predictions.append(TracePrediction(
                source=d["sentence1"],
                target=d["sentence2"],
                label=d["label"],
                score=score
            ))

        metrics = calculate_prediction_metrics(predictions)
        print(metrics)

        return metrics
