import numpy as np
from datasets import Dataset
from torch.utils.data import Sampler

from trace_model_trainer.constants import BATCH_SIZE


class BalancedSampler(Sampler):
    def __init__(self, dataset: Dataset, neg_sample_ratio: float = 1, batch_size=BATCH_SIZE):
        super().__init__(dataset)
        train_df = dataset.to_pandas()
        self.dataset = dataset
        self.pos_indices = train_df[train_df["label"] == 1].index.to_list()
        self.neg_indices = train_df[train_df["label"] == 0].index.to_list()
        self.neg_sample_ratio = neg_sample_ratio
        self.batch_size = batch_size
        print("Pos", len(self.pos_indices))
        print("Ratio: ", len(self.pos_indices) / len(self.neg_indices))

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
