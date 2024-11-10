from datasets import Dataset, tqdm

from trace_model_trainer.formatters.iformatter import IFormatter
from trace_model_trainer.tdata.trace_dataset import TraceDataset
from trace_model_trainer.transforms.augmentation import get_tfidf_important_phrase, split


class PhraseFormatter(IFormatter):
    def format(self, dataset: TraceDataset) -> Dataset:
        texts = list(dataset.artifact_map.values())
        text2phrase, corpus_important_words = get_tfidf_important_phrase(texts)

        text1 = []
        text2 = []
        labels = []
        for text in tqdm(texts, desc="Augmenting dataset samples"):
            t_important_phrase, t_important_words = text2phrase[text]

            text1.append(text)
            text2.append(t_important_phrase)
            labels.append(1)

            common_words = [w for w in split(text) if w.lower() not in t_important_words]

            for word in common_words:
                text1.append(text)
                text2.append(word)
                labels.append(0.1)

        return Dataset.from_dict({
            "text1": text1,
            "text2": text2,
            "label": labels
        })
