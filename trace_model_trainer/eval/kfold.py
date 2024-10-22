from typing import List

from trace_model_trainer.eval.splitters.isplitter import ISplitter
from trace_model_trainer.readers.trace_dataset import TraceDataset


def kfold(dataset: TraceDataset,
          split_sizes: List[float],
          splitter: ISplitter,
          n_folds: int) -> List[TraceDataset]:
    for fold_index in range(n_folds):
        splits = _create_splits(dataset, split_sizes, splitter)
        yield splits


def _create_splits(dataset: TraceDataset, split_sizes: List[float], splitter: ISplitter) -> List[TraceDataset]:
    assert sum(split_sizes) == 1, f"Sum of split does not equal 1."
    left = 1.0
    splits = []
    for split_index in range(len(split_sizes)):
        req = split_sizes[split_index]
        test_size = req / left
        if round(test_size, 3) == 1:
            splits.append(dataset)
        else:
            curr, dataset = splitter.split(dataset, test_size)
            splits.append(curr)
            left = left - req

    return splits
