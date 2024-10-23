from collections import defaultdict
from typing import Dict, List, Tuple

from pandas import DataFrame
from sklearn.model_selection import train_test_split

from trace_model_trainer.eval.splitters.isplitter import ISplitter
from trace_model_trainer.eval.trace_iterator import trace_iterator
from trace_model_trainer.tdata.trace_dataset import TraceDataset
from trace_model_trainer.tdata.types import TracePrediction
from trace_model_trainer.utils import create_source2targets


class QuerySplitter(ISplitter):
    def __init__(self, group_col: str = "target"):
        self.group_cop = group_col

    def split(self, dataset: TraceDataset, train_size: float, random_seed: int = None) -> Tuple[TraceDataset, TraceDataset]:
        """
        Splits trace dataset so that entire artifacts are removed from the training set and put into the test set.
        :param dataset: The dataset to split.
        :param train_size: The size (in percentage) of the training set.
        TODO: How should splitting be quelt between layers? ->
        DESIGN:
        :return: Training set followed by testing set.
        """
        query2items = defaultdict(list)
        for source_ids, target_ids in trace_iterator(dataset):
            for t_id in target_ids:
                query2items[t_id].extend(source_ids)

        train_queries, test_queries = train_test_split(list(query2items.keys()), train_size=train_size, random_state=random_seed)

        source2targets = create_source2targets(dataset.trace_df)

        train_df = self.extract_trace_df_from_query_map(source2targets, {q: query2items[q] for q in train_queries})
        test_df = self.extract_trace_df_from_query_map(source2targets, {q: query2items[q] for q in test_queries})

        train_trace_df = TraceDataset(
            artifact_df=dataset.artifact_df[~dataset.artifact_df["id"].isin(test_queries)],
            trace_df=train_df,
            layer_df=dataset.layer_df
        )
        test_trace_df = TraceDataset(
            artifact_df=dataset.artifact_df[~dataset.artifact_df["id"].isin(train_queries)],
            trace_df=test_df,
            layer_df=dataset.layer_df
        )
        return train_trace_df, test_trace_df

    @staticmethod
    def extract_trace_df_from_query_map(source2targets: Dict[str, Dict[str, int]], query_map: Dict[str, List[str]]):
        samples = []
        for query, query_items in query_map.items():
            for item in query_items:
                label = source2targets[item].get(query, 0)
                samples.append(TracePrediction(
                    source=item,
                    target=query,
                    label=label,
                    score=None  # Override score if one exists in trace df.
                ).to_json())
        train_df = DataFrame(samples)
        return train_df
