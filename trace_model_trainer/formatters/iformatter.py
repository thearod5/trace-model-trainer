from abc import ABC, abstractmethod

from datasets import Dataset

from trace_model_trainer.readers.trace_dataset import TraceDataset


class IFormatter(ABC):
    """
    Formats a dataset to match an expected loss function input.
    """

    @abstractmethod
    def format(self, dataset: TraceDataset) -> Dataset:
        raise NotImplementedError("Format method has not been implemented.")
