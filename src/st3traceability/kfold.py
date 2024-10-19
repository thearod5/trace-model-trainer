from typing import List, Tuple

from datasets import Dataset

from st3traceability.splitters.isplitter import ISplitter


def kfold(dataset: Dataset,
          split_sizes: Tuple[float, float, float],
          splitter: ISplitter,
          n_folds: int) -> List[Tuple[Dataset, Dataset, Dataset]]:
    for fold_index in range(n_folds):
        splits = _create_splits(dataset, split_sizes, splitter)
        yield splits


def _create_splits(dataset: Dataset, split_sizes: Tuple[float, float, float], splitter: ISplitter):
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
