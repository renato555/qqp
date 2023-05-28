import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class QQPDataset(Dataset):
    def __init__(self, csv_file, embeddings_file):
        self.embeddings = np.load(embeddings_file)
        self.csv_file = csv_file
        pair_ids, labels = self._load_ids_labels()
        self.pair_ids = pair_ids
        self.labels = labels

    def _load_ids_labels(self):
        df = pd.read_csv(self.csv_file)
        return df['pair_id'].values, df['is_duplicate'].values

    def __len__(self):
        return len(self.pair_ids)

    def __getitem__(self, idx):
        pair_id = self.pair_ids[idx]
        label = self.labels[idx]
        embedding = self.embeddings[pair_id]
        return embedding, label



if __name__ == '__main__':
    # Example usage
    csv_file = './data/a.csv'
    embeddings_file = './embeddings/all_embeddings.npy'

    dataset = QQPDataset(csv_file, embeddings_file)
    print(f'len(dataset): {len(dataset)}')
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    for x, y in dataloader:
        # Process the batch of embeddings
        print(x.shape)  # Example: prints (32, D) for a batch of size 32
        print(y.shape)
        print()
