from datasets import Dataset

from trace_model_trainer.eval.trace_iterator import trace_iterator_labeled
from trace_model_trainer.formatters.iformatter import IFormatter
from trace_model_trainer.tdata.trace_dataset import TraceDataset


class KatFormatter(IFormatter):
    def format(self, dataset: TraceDataset, artifact_df, pos_neg_ratio=.5) -> Dataset:
        artifact_map = dataset.artifact_map

        pos_indices = []
        neg_indices = []
        texts1 = []
        texts2 = []
        labels = []
        for i, (s_id, t_id, label) in enumerate(trace_iterator_labeled(dataset)):
            texts1.append(artifact_map[s_id])
            texts2.append(artifact_map[t_id])
            label = label * 0.5
            labels.append(label)

        for artifact_body in artifact_df["content"]:
            texts1.append(artifact_body)
            texts2.append(artifact_body)
            labels.append(1)

        return Dataset.from_dict({"text1": texts1, "text2": texts2, "label": labels}).shuffle()
