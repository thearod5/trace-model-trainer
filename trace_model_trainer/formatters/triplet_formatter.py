from collections import defaultdict

import pandas as pd
from datasets import Dataset

from trace_model_trainer.eval.trace_iterator import trace_iterator
from trace_model_trainer.formatters.iformatter import IFormatter
from trace_model_trainer.readers.trace_dataset import TraceDataset
from trace_model_trainer.utils import create_source2targets


class TripletFormatter(IFormatter):
    def format(self, dataset: TraceDataset) -> Dataset:
        artifact_map = dataset.artifact_map

        triplet_lookup = self.create_triplet_lookup(dataset)

        # Create triplets
        anchors = []
        positives = []
        negatives = []

        for source_id, lookup in triplet_lookup.items():
            positive_links = pd.Series(lookup['pos'])
            negative_links = pd.Series(lookup['neg']).sample(frac=1)
            for positive_id, negative_id in zip(positive_links, negative_links):
                anchors.append(artifact_map[source_id])
                positives.append(artifact_map[positive_id])
                negatives.append(artifact_map[negative_id])

        return Dataset.from_dict({
            "anchor": anchors,
            "positive": positives,
            "negative": negatives
        })

    @staticmethod
    def create_triplet_lookup(dataset: TraceDataset):
        lookup = defaultdict(lambda: {"pos": [], 'neg': []})
        trace_map = create_source2targets(dataset.trace_df)
        for source_artifact_ids, target_artifact_ids in trace_iterator(dataset):
            for s_id in source_artifact_ids:
                for t_id in target_artifact_ids:
                    label = trace_map[s_id].get(t_id, 0)
                    if label == 0:
                        lookup[t_id]['neg'].append(s_id)
                    else:
                        lookup[t_id]['pos'].append(s_id)

        return lookup
