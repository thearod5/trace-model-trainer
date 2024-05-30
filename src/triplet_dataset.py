from torch.utils.data import Dataset


class TripletDataset(Dataset):
    def __init__(self, triplet_df):
        self.triplet_df = triplet_df

    def __len__(self):
        return len(self.triplet_df)

    def __getitem__(self, idx):
        row = self.triplet_df.iloc[idx]
        return row
