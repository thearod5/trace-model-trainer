import numpy as np
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
            (pos_indices if label == 1 else neg_indices).append(i)

        for artifact_body in artifact_df["content"]:
            texts1.append(artifact_body)
            texts2.append(artifact_body)
            labels.append(1)
            pos_indices.append(len(labels) - 1)

        n_neg = round(len(pos_indices) * (1 / pos_neg_ratio))
        sampled_neg_indices = np.random.choice(neg_indices, n_neg, replace=True)
        combined_indices = np.concatenate([pos_indices, sampled_neg_indices])
        np.random.shuffle(combined_indices)
        indices = combined_indices.tolist()

        final_text1 = []
        final_text2 = []
        final_labels = []

        for i in indices:
            final_text1.append(texts1[i])
            final_text2.append(texts2[i])
            final_labels.append(labels[i])

        return Dataset.from_dict({"text1": final_text1, "text2": final_text2, "label": final_labels}).shuffle()
