from datasets import Dataset

from trace_model_trainer.formatters.iformatter import IFormatter
from trace_model_trainer.tdata.trace_dataset import TraceDataset


class ArtifactAugmentationFormatter(IFormatter):

    def format(self, dataset: TraceDataset) -> Dataset:
        return Dataset.from_dict({"texts": dataset.artifact_df["content"].tolist()})
