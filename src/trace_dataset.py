from torch.utils.data import Dataset


class TraceDataset(Dataset):
    def __init__(self, df, artifact_map):
        self.df = df
        self.artifact_map = artifact_map

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return row["s_id"], row["t_id"], row['label']

    def get_prediction_payload(self):
        payload = []
        for s, t, l in self:
            payload.append((self.artifact_map[s], self.artifact_map[t]))
        return payload
