from datasets import Dataset

from readers.trace_dataset import TraceDataset
from formatters.iformatter import IFormatter


class ClassificationFormatter(IFormatter):
    """
    Formats dataset as list of classified pairs of text (source/target) as either traced (1) or not traced (0).
    """

    def format(self, dataset: TraceDataset) -> Dataset:
        artifact_map = dataset.artifact_map

        sources = []
        targets = []
        labels = []
        for i, trace in dataset.trace_df.iterrows():
            sources.append(artifact_map[trace["source"]])
            targets.append(artifact_map[trace["target"]])
            labels.append(trace.get("label", 1))
        return Dataset.from_dict({"sentence1": sources, "sentence2": targets, "label": labels})
