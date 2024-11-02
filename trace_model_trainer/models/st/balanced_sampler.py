import numpy as np
from datasets import Dataset
from torch.utils.data import ConcatDataset, Sampler


class BalancedSampler(Sampler):
    def __init__(self, dataset: Dataset, batch_size: int, neg_sample_ratio: float = 2):
        assert batch_size is not None
        super().__init__(dataset)
        self.pos_indices, self.neg_indices = self.extract_indices(dataset)
        self.dataset = dataset
        self.neg_sample_ratio = neg_sample_ratio
        self.batch_size = batch_size
        print("Using balanced sampler")

    @staticmethod
    def extract_indices(dataset: Dataset | ConcatDataset):
        if isinstance(dataset, Dataset):
            df = dataset.to_pandas()
            pos_indices = df[df["label"] == 1].index.to_list()
            neg_indices = df[df["label"] == 0].index.to_list()
        else:
            pos_indices = []
            neg_indices = []
            for i, item in enumerate(dataset):
                (pos_indices if item["label"] == 1 else neg_indices).append(i)

        return pos_indices, neg_indices

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
        Creates balanced batches with the specified neg_sample_ratio and shuffles them at the end.
        :return: List of batches (each batch is a list of indices)
        """
        # Calculate the number of positive and negative samples per batch
        n_pos_per_batch = self.batch_size // (self.neg_sample_ratio + 1)
        n_neg_per_batch = self.batch_size - n_pos_per_batch  # Rest of the batch should be negative

        # Shuffle positive and negative indices
        np.random.shuffle(self.pos_indices)
        np.random.shuffle(self.neg_indices)

        # Replicate negative indices to ensure there are enough to match the ratio
        n_neg_needed = n_neg_per_batch * (len(self.pos_indices) // n_pos_per_batch + 1)
        sampled_neg_indices = np.random.choice(self.neg_indices, n_neg_needed, replace=True).tolist()

        pos_pointer = 0
        neg_pointer = 0
        batches = []

        # Create batches by combining positive and negative samples
        while pos_pointer < len(self.pos_indices):
            # Get the next set of positive and negative samples
            pos_batch = self.pos_indices[pos_pointer: pos_pointer + n_pos_per_batch]
            neg_batch = sampled_neg_indices[neg_pointer: neg_pointer + n_neg_per_batch]

            # Combine and shuffle the batch
            batch = pos_batch + neg_batch
            np.random.shuffle(batch)  # Shuffle within the batch to mix positives and negatives

            # Check if the batch size is correct
            if len(batch) == self.batch_size:
                batches.append(batch)

            pos_pointer += n_pos_per_batch
            neg_pointer += n_neg_per_batch

        # Shuffle all batches at the end
        np.random.shuffle(batches)
        return batches
