from datasets import Dataset
from torch.utils.data import Sampler


class AugmentedSampler(Sampler):
    def __init__(self, dataset: Dataset):
        super().__init__(dataset)
        self.batches = self.create_batches()

    def __iter__(self):
        """
        Create iterator over balanced training data.
        :return: Iterator of training batches.
        """
        for batch in self.batches:
            yield batch
        self.batches = self.create_batches()

    def create_batches(self):
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

        for curr in artifact_texts:
            text_words = [w.strip().lower() for w in curr.split()]
            text_common_words = list(set(common_words).intersection(set(text_words)))

            for i in range(n_per_text):
                words_to_remove = np.random.choice(text_common_words, 3, replace=False)
                text1.append(curr)
                text2.append(remove_common_words(curr, remove_words=words_to_remove))
                labels.append(1)
            for other in artifact_texts:
                if curr == 0:
                    continue
                text1.append(curr)
                text2.append(other)
                labels.append(0)

        # Step 5: Format the dataset
        return Dataset.from_dict({
            "text1": text1,
            "text2": text2,
            "label": labels
        })
