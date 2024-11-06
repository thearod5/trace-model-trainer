from collections import Counter
from typing import List

import numpy as np
from datasets import Dataset

from trace_model_trainer.formatters.iformatter import IFormatter
from trace_model_trainer.tdata.trace_dataset import TraceDataset


class ArtifactAugmentationFormatter(IFormatter):
    def format(self, dataset: TraceDataset) -> Dataset:
        # Step 1: Extract all artifact content from the dataset
        artifact_texts = list(dataset.artifact_df["content"])

        word2count = {}
        for a_text in artifact_texts:
            for word in a_text.split():
                word_id = word.lower()
                if word_id not in word2count:
                    word2count[word_id] = 0
                word2count[word_id] += 1

        common_word_map = list(sorted(word2count.items(), key=lambda t: t[1], reverse=True))[:30]
        common_words = [w[0] for w in common_word_map]

        # Helper function to remove common words from a sentence
        def remove_common_words(body, remove_words):
            return ' '.join([w for w in body.split() if w.lower() not in remove_words])

        # Step 3: Create augmented positive examples
        n_per_text = 3

        text1 = []
        text2 = []
        labels = []

        for text in artifact_texts:
            text_words = [w.strip().lower() for w in text.split()]
            text_common_words = list(set(common_words).intersection(set(text_words)))

            for i in range(n_per_text):
                words_to_remove = np.random.choice(text_common_words, 3, replace=False)
                text1.append(text)
                text2.append(remove_common_words(text, remove_words=words_to_remove))
                labels.append(1)
            # sample artifact_texts
            others = np.random.choice(artifact_texts, n_per_text, replace=False)
            for o in others:
                if text == 0:
                    continue
                text1.append(text)
                text2.append(o)
                labels.append(0)

        # Step 5: Format the dataset
        return Dataset.from_dict({
            "text1": text1,
            "text2": text2,
            "label": labels
        })


def generate_ngraphs(sentences: List[str]):
    n = 2  # For bigrams; change to 3 for trigrams, etc.
    all_ngrams = []
    for sentence in sentences:
        tokens = sentence.split()
        ngrams = generate_ngrams(tokens, n)
        all_ngrams.extend(ngrams)
    ngram_counts = Counter(all_ngrams)
    most_common_ngrams = ngram_counts.most_common()

    return most_common_ngrams


def generate_ngrams(tokens, n):
    return zip(*[tokens[i:] for i in range(n)])
