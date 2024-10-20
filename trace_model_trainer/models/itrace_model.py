from abc import ABC, abstractmethod
from typing import Dict, List

from readers.trace_dataset import TraceDataset

SimilarityMatrix = List[List[float]]


class ITraceModel(ABC):
    @abstractmethod
    def train(self, dataset_map: Dict[str, TraceDataset], *args, **kwargs):
        """
        Trains model on dataset.
        :param dataset_map: The dataset to tune model to.
        :param args: Positional arguments to pass in.
        :param kwargs: Keyword arguments to pass.
        :return: None. Model modified in place.
        """
        raise NotImplementedError()

    @abstractmethod
    def predict(self, sources: List[str], targets: List[str]) -> SimilarityMatrix:
        """
        Generate predictions between sources and targets.
        :param sources: List of artifacts referenced by "source" prediction tag.
        :param targets: List of artifacts referenced by "target" prediction tag
        :return: List of trace predictions.
        """
        raise NotImplementedError()

    def predict_single(self, source: str, target: str) -> float:
        """
        Generates a single prediction between source and target text.
        :param source: The source text.
        :param target: The target text.
        :return: A trace prediction.
        """
        return self.predict([source], [target])[0][0]
