from pandas import DataFrame

from trace_model_trainer.readers.utils import get_content
from trace_model_trainer.utils import t_id_creator

""""
ArtifactDataFrame: (id, content, summary)
TraceDataFrame: (source,target,label)
"""
ArtifactDataFrame = DataFrame
TraceDataFrame = DataFrame


class TraceDataset:
    def __init__(self, artifact_df: DataFrame, trace_df: DataFrame, layer_df: DataFrame):
        """
        Creates traceability dataset.
        :param artifact_df: DataFrame containing artifacts (id,content,summary,layer)
        :param trace_df: DataFrame containing trace links between artifacts.
        :param layer_df: DataFrame containing the types being traced (Artifact.layer)
        """
        self.artifact_df = artifact_df
        self.trace_df = filter_referenced_artifacts(trace_df, artifact_df)
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

    def get_layer_iterator(self, empty_ok: bool = False):
        if len(self.layer_df) == 0 and not empty_ok:
            raise Exception("Attempted to retrieve combination of traces, but no matrix defined.")

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


def filter_referenced_artifacts(trace_df, artifact_df):
    artifacts = set(artifact_df["id"].unique())
    return trace_df[trace_df["source"].isin(artifacts) & trace_df["target"].isin(artifacts)]
