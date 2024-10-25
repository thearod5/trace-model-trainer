import numpy as np
from datasets import Dataset
from torch.utils.data import Sampler


class BalancedSampler(Sampler):
    def __init__(self, dataset: Dataset, batch_size: int, neg_sample_ratio: float = 1):
        assert batch_size is not None
        super().__init__(dataset)
        df = dataset.to_pandas()
        self.dataset = dataset
        self.pos_indices = df[df["label"] == 1].index.to_list()
        self.neg_indices = df[df["label"] == 0].index.to_list()
        self.neg_sample_ratio = neg_sample_ratio
        self.batch_size = batch_size

    def __iter__(self):
        """
        Create iterator over balanced training data.
        :return: Iterator of training batches.
        """
        for batch in self.create_batches():
            yield batch

    def __len__(self):
        """
        Calculates the expected number of balanced batches.
        :return: Number of batches.
        """
        n_samples = len(self.pos_indices) + (len(self.pos_indices) * self.neg_sample_ratio)
        n_batches = n_samples // self.batch_size
        if n_samples % self.batch_size != 0:
            n_batches += 1
        return n_batches

    def create_batches(self):
        """
        Randomly down samples negatives to create balanced set of batches.
        :return: List of batches (indices)
        """
        n_neg = int(len(self.pos_indices) * self.neg_sample_ratio)

        sampled_neg_indices = np.random.choice(self.neg_indices, n_neg, replace=True)

        # Combine positive and negative indices and shuffle
        combined_indices = np.concatenate([self.pos_indices, sampled_neg_indices])
        np.random.shuffle(combined_indices)
        indices = combined_indices.tolist()

        batches = [indices[i:i + self.batch_size] for i in range(0, len(indices), self.batch_size)]
        return batches
