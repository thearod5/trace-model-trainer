from pandas import DataFrame

from trace_model_trainer.readers.types import Artifact
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
        assert all(c in artifact_df.columns for c in ["id", "content", "summary", "layer"]), f"Result: {artifact_df.columns}"
        assert all(c in trace_df.columns for c in ["source", "target"]), f"Result: {trace_df.columns}"
        assert all(c in layer_df.columns for c in ["source_type", "target_type"]), f"Result: {layer_df.columns}"

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

    def get_by_type(self, type_name: str) -> ArtifactDataFrame:
        return self.artifact_df[self.artifact_df['layer'] == type_name]


def filter_referenced_artifacts(trace_df, artifact_df):
    artifacts = set(artifact_df["id"].unique())
    return trace_df[trace_df["source"].isin(artifacts) & trace_df["target"].isin(artifacts)]


def get_content(a: Artifact):
    """
    Returns summary of artifacts if exists otherwise returns its content.
    :param a: Artifact to extract content for.
    :return: Artifact content.
    """
    s = a.get("summary", None)
    if isinstance(s, str) and len(s) > 0:
        return s
    return a["content"]
