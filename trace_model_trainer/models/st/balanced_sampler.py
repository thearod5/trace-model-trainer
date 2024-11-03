import numpy as np
from datasets import Dataset
from torch.utils.data import ConcatDataset, Sampler


class BalancedSampler(Sampler):
    def __init__(self, dataset: Dataset, batch_size: int, neg_sample_ratio: float = 3):
        print("Using BalancedSampler")
        assert batch_size is not None
        assert batch_size % (neg_sample_ratio + 1) == 0, f"Batch size ({batch_size}) must be divisible by {neg_sample_ratio + 1}"
        super().__init__(dataset)
        self.pos_indices, self.neg_indices = self.extract_indices(dataset)
        self.dataset = dataset
        self.neg_sample_ratio = neg_sample_ratio
        self.batch_size = batch_size

        assert len(self.pos_indices) > 0, f"Received samples with no positive indices"
        assert len(self.neg_indices) > 0, f"Received samples with no negative indices"

        self.batches = self.create_batches()

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
        for batch in self.batches:
            yield batch
        self.batches = self.create_batches()

    def __len__(self):
        """
        Calculates the expected number of balanced batches.
        :return: Number of batches.
        """
        return len(self.batches)

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
                print("Batch does not contain enough samples...")
            batches.append(batch)

            pos_pointer += n_pos_per_batch
            neg_pointer += n_neg_per_batch

        # Shuffle all batches at the end
        np.random.shuffle(batches)
        assert len(batches) > 0, "No batches resulted from BalancedSampler."
        assert len(batches[0]) > 0, f"An empty batch was found in batches ({batches})"
        return batches
