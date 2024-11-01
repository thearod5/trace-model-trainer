from datasets import Dataset

from trace_model_trainer.eval.trace_iterator import trace_iterator_labeled
from trace_model_trainer.formatters.iformatter import IFormatter
from trace_model_trainer.tdata.trace_dataset import TraceDataset


class KatFormatter(IFormatter):
    def format(self, dataset: TraceDataset) -> Dataset:
        artifact_map = dataset.artifact_map

        texts1 = []
        texts2 = []
        labels = []
        for s_id, t_id, label in trace_iterator_labeled(dataset):
            texts1.append(artifact_map[s_id])
            texts2.append(artifact_map[t_id])
            label = label * 0.5
            labels.append(label)

        for artifact_id, artifact_body in artifact_map.items():
            texts1.append(artifact_body)
            texts2.append(artifact_body)
            labels.append(1)

        return Dataset.from_dict({"text1": texts1, "text2": texts2, "label": labels}).shuffle()
