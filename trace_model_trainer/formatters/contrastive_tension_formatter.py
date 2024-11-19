from typing import List

from datasets import Dataset

from trace_model_trainer.formatters.iformatter import IFormatter
from trace_model_trainer.tdata.trace_dataset import TraceDataset


class ContrastiveTensionFormatter(IFormatter):
    def format(self, dataset: TraceDataset) -> Dataset:
        """
        Formats a dataset for training on just artifacts (sentences).
        :param dataset: The dataset whose artifacts are formatted.
        :return: The dataset.
        """
        train_examples: List[str] = list(dataset.artifact_map.values())

        text1 = []
        text2 = []
        labels = []

        for example in train_examples:
            for other in train_examples:
                text1.append(example)
                text2.append(other)
                label = 1 if example == other else 0
                labels.append(label)

        return Dataset.from_dict({
            "sentence1": text1,
            "sentence2": text2,
            "label": labels
        })
