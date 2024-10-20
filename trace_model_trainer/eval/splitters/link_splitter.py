from typing import Tuple

from sklearn.model_selection import train_test_split

from eval.splitters.isplitter import ISplitter
from readers.trace_dataset import ArtifactDataFrame, TraceDataFrame, TraceDataset


class LinkSplitter(ISplitter):
    def split(self, dataset: TraceDataset, train_size: float) -> Tuple[TraceDataset, TraceDataset]:
        """
        Splits dataset into segments of given sizes, stratifying the labels to enable equal proportions of true links.
        :param dataset: The dataset to split.
        :param train_size: The percentage of the data to keep in the first split.
        :return: Train split and test split.
        """
        df = dataset.trace_df
        train_df, test_df = train_test_split(df, stratify=df["label"], test_size=train_size)

        train_trace_dataset = TraceDataset(
            artifact_df=self.filter_unreferenced_artifacts(train_df, dataset.artifact_df),
            trace_df=train_df,
            layer_df=dataset.layer_df
        )

        test_trace_dataset = TraceDataset(
            artifact_df=self.filter_unreferenced_artifacts(test_df, dataset.artifact_df),
            trace_df=test_df,
            layer_df=dataset.layer_df
        )

        return train_trace_dataset, test_trace_dataset

    @staticmethod
    def filter_unreferenced_artifacts(trace_df: TraceDataFrame, artifact_df: ArtifactDataFrame) -> ArtifactDataFrame:
        """
        Removes unreferenced artifacts from data frame.
        :param trace_df:
        :param artifact_df: The artifacts in the system.
        :return: Dataframe containing only artifacts referenced in trace links.
        """
        referenced_artifacts = set(trace_df["source"]).union(set(trace_df["target"]))
        artifact_df = artifact_df[artifact_df["id"].isin(referenced_artifacts)]
        return artifact_df
