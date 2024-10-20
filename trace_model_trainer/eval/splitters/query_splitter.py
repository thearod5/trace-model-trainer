from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from eval.splitters.isplitter import ISplitter
from readers.trace_dataset import TraceDataset


class QuerySplitter(ISplitter):
    def __init__(self, group_col: str):
        self.group_cop = group_col

    def split(self, dataset: TraceDataset, train_size: float) -> Tuple[TraceDataset, TraceDataset]:
        """
        Splits trace dataset so that entire artifacts are removed from the training set and put into the test set.
        :param dataset: The dataset to split.
        :param train_size: The size (in percentage) of the training set.
        :return: Training set followed by testing set.
        """
        df = dataset.trace_df

        group_map = {group_name: group for group_name, group in df.groupby(self.group_cop)}
        train_groups, test_groups = train_test_split(list(group_map.keys()), train_size=train_size)

        train_df = pd.concat([group_map[g] for g in train_groups])
        test_df = pd.concat([group_map[g] for g in test_groups])

        print("Test Groups:", test_groups)

        train_trace_df = TraceDataset(
            artifact_df=dataset.artifact_df[~dataset.artifact_df["id"].isin(test_groups)],
            trace_df=train_df,
            layer_df=dataset.layer_df
        )
        test_trace_df = TraceDataset(
            artifact_df=dataset.artifact_df[~dataset.artifact_df["id"].isin(train_groups)],
            trace_df=test_df,
            layer_df=dataset.layer_df
        )
        return train_trace_df, test_trace_df
