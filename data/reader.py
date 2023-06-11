import csv
import os

from torch.utils import data

class Dataset(data.Dataset):
    def __init__(self, split, data_dir, embeddings_kv):
        split_map = {
            "train": "train.csv",
            "val": "val.csv",
            "test": "test.csv"
        }
        filename = split_map[split]
        
        self.xs = []
        self.ys = []
        
        with open(os.path.join(data_dir, filename)) as f:
            reader = csv.reader(f)
            for row in reader:
                self.xs.append([embeddings_kv.key_to_index[w] for w in row[1].split(' ') if w in embeddings_kv])
                self.ys.append(int(row[0] == "spam"))
    
    def __len__(self):
        return len(self.xs)
    
    def __getitem__(self, idx):
        return self.xs[idx], self.ys[idx]
