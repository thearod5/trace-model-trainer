from datasets import Dataset
from sentence_transformers.losses import ContrastiveTensionDataLoader

from readers.trace_dataset import TraceDataset
from formatters.iformatter import IFormatter


class ContrastiveTensionFormatter(IFormatter):
    def format(self, dataset: TraceDataset) -> Dataset:
        train_examples = list(dataset.artifact_df["content"])
        examples = [e for e in ContrastiveTensionDataLoader(train_examples, batch_size=6, pos_neg_ratio=3)]

        text1 = [e.texts[0] for e_batch in examples for e in e_batch]
        text2 = [e.texts[1] for e_batch in examples for e in e_batch]
        labels = [e.label for e_batch in examples for e in e_batch]
        return Dataset.from_dict({
            "text1": text1,
            "texts2": text2,
            "label": labels
        })
