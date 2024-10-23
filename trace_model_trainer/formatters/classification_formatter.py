from datasets import Dataset

from trace_model_trainer.formatters.iformatter import IFormatter
from trace_model_trainer.tdata.trace_dataset import TraceDataset

DEFAULT_KEY_MAP = {"sentence1": "source", "sentence2": "target", "label": "label"}


class ClassificationFormatter(IFormatter):
    """
    Formats dataset as list of classified pairs of text (source/target) as either traced (1) or not traced (0).
    """

    def __init__(self, key_map: str = None):
        key_map = key_map or DEFAULT_KEY_MAP
        assert all([k in key_map for k in ["sentence1", "sentence2", "label"]])
        self.key_map = key_map

    def format(self, dataset: TraceDataset) -> Dataset:
        artifact_map = dataset.artifact_map

        texts1 = []
        texts2 = []
        labels = []
        for i, trace in dataset.trace_df.iterrows():
            texts1.append(artifact_map[trace[self.key_map["sentence1"]]])
            texts2.append(artifact_map[trace[self.key_map["sentence2"]]])
            labels.append(trace.get(self.key_map["label"], 1))
        return Dataset.from_dict({"sentence1": texts1, "sentence2": texts2, "label": labels})
