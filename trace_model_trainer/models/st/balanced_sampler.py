import numpy as np
from datasets import Dataset
from torch.utils.data import ConcatDataset, Sampler


class BalancedSampler(Sampler):
    def __init__(self, dataset: Dataset, batch_size: int, neg_sample_ratio: float = 4, resample_data: bool = True):
        print("Using BalancedSampler")
        super().__init__(dataset)
        self.pos_indices, self.neg_indices = self.extract_indices(dataset)
        self.dataset = dataset
        self.neg_sample_ratio = neg_sample_ratio
        self.batch_size = batch_size
        self.resample_data = resample_data

        assert len(self.pos_indices) > 0, f"Received samples with no positive indices"
        assert len(self.neg_indices) > 0, f"Received samples with no negative indices"

        self.batches = self.create_batches()

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
        neg_indices = self.neg_indices
        if self.resample_data:
            # Select negative links to use in batches
            n_negative = int(self.neg_sample_ratio * len(self.pos_indices))
            neg_indices = np.random.choice(self.neg_indices, min(n_negative, len(self.neg_indices)), replace=False).tolist()
            # labels = [self.dataset[i]["label"] for i in neg_indices + self.pos_indices]
            # print("Batched Training Data:\n", pd.Series(labels).value_counts())

        # Create batches
        indices = neg_indices + self.pos_indices
        np.random.shuffle(indices)
        batches = [indices[i:i + self.batch_size] for i in range(0, len(indices), self.batch_size)]

        assert len(batches) > 0, "No batches resulted from BalancedSampler."
        assert len(batches[0]) > 0, f"An empty batch was found in batches ({batches})"

        return batches

    @staticmethod
    def extract_indices(dataset: Dataset | ConcatDataset):
        if isinstance(dataset, Dataset):
            df = dataset.to_pandas()
            pos_indices = df[df["label"] >= 0.5].index.to_list()
            neg_indices = df[df["label"] < 0.5].index.to_list()
        else:
            pos_indices = []
            neg_indices = []
            for i, item in enumerate(dataset):
                score = item["label"] if "label" in item else item["score"]
                (pos_indices if score >= 0.5 else neg_indices).append(i)

        return pos_indices, neg_indices
