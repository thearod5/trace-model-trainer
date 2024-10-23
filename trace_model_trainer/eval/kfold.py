import random
from typing import List

from trace_model_trainer.eval.splitters.isplitter import ISplitter
from trace_model_trainer.tdata.trace_dataset import TraceDataset


def kfold(dataset: TraceDataset,
          split_sizes: List[float],
          splitter: ISplitter,
          n_folds: int,
          random_seeds: List[int] = None) -> List[TraceDataset]:
    for fold_index in range(n_folds):
        random_seed = random_seeds[fold_index] if random_seeds else random.randint(1, 10000)
        splits = _create_splits(dataset, split_sizes, splitter, random_seed=random_seed)
        yield splits


def _create_splits(dataset: TraceDataset, split_sizes: List[float], splitter: ISplitter, **kwargs) -> List[TraceDataset]:
    assert sum(split_sizes) == 1, f"Sum of split does not equal 1."
    left = 1.0
    splits = []
    for split_index in range(len(split_sizes)):
        req = split_sizes[split_index]
        test_size = req / left
        if round(test_size, 3) == 1:
            splits.append(dataset)
        else:
            curr, dataset = splitter.split(dataset, test_size, **kwargs)
            splits.append(curr)
            left = left - req

    return splits
