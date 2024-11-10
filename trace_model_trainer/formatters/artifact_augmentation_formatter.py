from datasets import Dataset, DatasetDict

from trace_model_trainer.formatters.iformatter import IFormatter
from trace_model_trainer.tdata.trace_dataset import TraceDataset


class ArtifactAugmentationFormatter(IFormatter):

    def format(self, dataset: TraceDataset) -> Dataset:
        return DatasetDict({
            "artifact_map": dataset.artifact_map,
            "traces": Dataset.from_dict(dataset.trace_df)
        })
