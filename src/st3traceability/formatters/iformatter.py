from abc import ABC, abstractmethod

from datasets import Dataset


class IFormatter(ABC):
    """
    Formats a dataset to match an expected loss function input.
    """
    @abstractmethod
    def format(self, dataset: Dataset) -> Dataset:
        raise NotImplementedError("Format method has not been implemented.")