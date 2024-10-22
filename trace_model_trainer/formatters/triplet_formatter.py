from collections import defaultdict

import pandas as pd
from datasets import Dataset

from trace_model_trainer.formatters.iformatter import IFormatter
from trace_model_trainer.readers.trace_dataset import TraceDataset
from trace_model_trainer.utils import t_id_creator


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
        for source_artifact_ids, target_artifact_ids in dataset.get_layer_iterator():
            for s_id in source_artifact_ids:
                for t_id in target_artifact_ids:
                    trace_id = t_id_creator(source=s_id, target=t_id)

                    if trace_id not in dataset.trace_map:
                        lookup[s_id]['neg'].append(t_id)
                    else:
                        lookup[s_id]['pos'].append(t_id)

        return lookup
