import os

import pandas as pd
import torch
from torch.utils import data

class Dataset(data.Dataset):
    def __init__(self, split, data_dir, embeddings_kv, length):
        xs = []
        ys = []
        df = pd.read_csv(os.path.join(data_dir, split, "yelp_ratings.csv"))
        for text in df['text']:
            ws = [
                embeddings_kv.key_to_index[w]
                for w in text.split(" ")
                if w in embeddings_kv
            ]
            # pad with dots (must pad for input to be accepted; since inputs must be of same size)
            ws += [embeddings_kv.key_to_index["."]] * (length - len(ws))
            xs.append(ws[:length])
        for label in df['sentiment']:
            ys.append(int(label == 1))
        self.xs = torch.LongTensor(xs)
        self.ys = torch.LongTensor(ys)

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        return self.xs[idx], self.ys[idx]
