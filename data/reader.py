import csv
import os

import torch
from torch.utils import data


class Dataset(data.Dataset):
    def __init__(self, split, data_dir, embeddings_kv, length):
        xs = []
        ys = []
        with open(os.path.join(data_dir, split, "spam.csv")) as f:
            reader = csv.reader(f)
            for row in reader:
                ws = [
                    embeddings_kv.key_to_index[w]
                    for w in row[1].split(" ")
                    if w in embeddings_kv
                ]
                # pad with dots (must pad for input to be accepted; since inputs must be of same size)
                ws += [embeddings_kv.key_to_index["."]] * (length - len(ws))
                xs.append(ws[:length])
                ys.append(int(row[0] == "spam"))
        self.xs = torch.LongTensor(xs)
        self.ys = torch.LongTensor(ys)

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        return self.xs[idx], self.ys[idx]
