from abc import ABC, abstractmethod
from typing import Tuple

from datasets import Dataset


class ISplitter(ABC):
    @abstractmethod
    def split(self, dataset: Dataset, split_size: float) -> Tuple[Dataset, Dataset]:
        """Splits dataset based on strategy of given size for test set."""
        raise NotImplementedError("`split` method is not implemented.")
