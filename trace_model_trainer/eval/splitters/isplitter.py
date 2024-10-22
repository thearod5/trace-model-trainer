from abc import ABC, abstractmethod
from typing import Tuple

from trace_model_trainer.readers.trace_dataset import TraceDataset


class ISplitter(ABC):
    @abstractmethod
    def split(self, dataset: TraceDataset, train_size: float, **kwargs) -> Tuple[TraceDataset, TraceDataset]:
        """
        Splits dataset based on strategy of given size for test set.
        :param dataset: The dataset to split.
        :param train_size: The size of the first split (typically training set).
        :return train / test splits with given proportions.
        """
        raise NotImplementedError("`split` method is not implemented.")
