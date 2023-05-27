import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class QQPDataset(Dataset):
    def __init__(self, csv_file, embeddings_folder):
        self.embeddings_folder = embeddings_folder
        self.embedding_files = sorted(os.listdir(self.embeddings_folder), key=lambda name: int(name.split(".")[0]))
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
        file_idx = pair_id // 1000
        embedding_idx = pair_id % 1000
        file_path = os.path.join(self.embeddings_folder, self.embedding_files[file_idx])
        embeddings = np.load(file_path)
        embedding = embeddings[embedding_idx]
        return embedding, label



if __name__ == '__main__':
    # Example usage
    csv_file = './data/a.csv'
    embeddings_folder = './embeddings'

    dataset = QQPDataset(csv_file, embeddings_folder)
    print(f'len(dataset): {len(dataset)}')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    for x, y in dataloader:
        # Process the batch of embeddings
        print(x.shape)  # Example: prints (32, D) for a batch of size 32
        print(y.shape)
        print()
