from typing import Tuple

import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split

from st3traceability.splitters.isplitter import ISplitter


class QuerySplitter(ISplitter):
    def __init__(self, group_col: str):
        self.group_cop = group_col

    def split(self, dataset: Dataset, split_size: float) -> Tuple[Dataset, Dataset]:
        df = dataset.to_pandas()

        group_map = {group_name: group for group_name, group in df.groupby(self.group_cop)}
        train_groups, test_groups = train_test_split(list(group_map.keys()), train_size=split_size)

        train_dataset = Dataset.from_dict(pd.concat([group_map[g] for g in train_groups]))
        test_dataset = Dataset.from_dict(pd.concat([group_map[g] for g in test_groups]))

        return train_dataset, test_dataset
