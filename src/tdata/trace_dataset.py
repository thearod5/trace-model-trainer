from typing import Tuple

from pandas import DataFrame
from sklearn.model_selection import train_test_split

from tdata.utils import get_artifact_ids_from_trace_df, get_content
from utils import t_id_creator

""""
ArtifactDataFrame: (id, content, summary)
TraceDataFrame: (source,target,label)
"""
ArtifactDataFrame = DataFrame
TraceDataFrame = DataFrame


class TraceDataset:
    def __init__(self, artifact_df: DataFrame, trace_df: DataFrame, layer_df: DataFrame):
        self.artifact_df = artifact_df
        self.trace_df = trace_df
        self.layer_df = layer_df
        self.artifact_map = {a['id']: get_content(a.to_dict()) for i, a in artifact_df.iterrows()}
        self.trace_map = {t_id_creator(r): r.to_dict() for i, r in trace_df.iterrows()}

    def __copy__(self):
        return TraceDataset(self.artifact_df.copy(), self.trace_df.copy(), self.layer_df.copy())

    def __len__(self):
        return len(self.trace_df)

    def __getitem__(self, idx):
        row = self.trace_df.iloc[idx]
        return row["source"], row["target"], row['label']

    def get_layer_iterator(self):
        payload = []
        for i, layer_row in self.layer_df.iterrows():
            source_layer = layer_row["source_type"]
            target_layer = layer_row["target_type"]

            source_artifact_df = self.get_by_type(source_layer)
            target_artifact_df = self.get_by_type(target_layer)

            source_artifact_ids = source_artifact_df['id']
            target_artifact_ids = target_artifact_df['id']
            payload.append((source_artifact_ids, target_artifact_ids))
        return payload

    def get_by_type(self, type_name: str) -> ArtifactDataFrame:
        return self.artifact_df[self.artifact_df['layer'] == type_name]

    def split(self, test_size: float) -> Tuple["TraceDataset", "TraceDataset"]:
        indices = list(range(len(self.trace_df)))
        a_trace_indices, b_trace_indices = train_test_split(indices, test_size=test_size, random_state=42)
        a_trace_df = self.trace_df.iloc[a_trace_indices].reset_index(drop=True)
        b_trace_df = self.trace_df.iloc[b_trace_indices].reset_index(drop=True)
        return self.subset(a_trace_df), self.subset(b_trace_df)

    def subset(self, trace_df: DataFrame):
        artifact_ids_set = get_artifact_ids_from_trace_df(trace_df)
        artifact2layer = {a['id']: a['layer'] for _, a in self.artifact_df.iterrows()}

        trace_entries = []
        layer_entries = set()
        for i, row in trace_df.iterrows():
            s_id = row['source']
            t_id = row['target']

            if s_id not in artifact_ids_set or t_id not in artifact_ids_set:
                continue

            trace_entries.append(row)

            s_layer = artifact2layer[s_id]
            t_layer = artifact2layer[t_id]
            layer_hash = f"{s_layer}*{t_layer}"
            layer_entries.add(layer_hash)

        layer_entries = [e.split("*") for e in layer_entries]

        artifact_df = self.artifact_df[self.artifact_df["id"].isin(artifact_ids_set)]
        trace_df = DataFrame(trace_entries)
        layer_df = DataFrame(layer_entries, columns=["source_type", "target_type"])
        return TraceDataset(artifact_df, trace_df, layer_df)
