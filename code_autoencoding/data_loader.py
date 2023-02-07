import pandas as pd
import numpy as np
import torch
from cmapPy.pandasGEXpress.parse import parse
from torch.utils.data import DataLoader, Dataset


def build_loaders(path, batch_size=128, split=0.9, in_format = "counts"):
    # Parse the gctx file, store as dataframe



    # Form the data into vectors. Default flags are for log2+1 and max_min scaling
    if (in_format.lower() == 'z_score'):
        vectors = vectorize(df, scale_min_max=False, scale_log2_1=False)
    else:
        vectors = vectorize(df)

    # Shuffle the vectors for random allocation to training and testing
    shuffled_vectors = shuffle(vectors)

    # Split into test and training
    split_idx = int(split * len(shuffled_vectors))
    train_set = TorchVectors(shuffled_vectors[:split_idx])
    test_set = TorchVectors(shuffled_vectors[split_idx:])

    # Construct loaders
    train_loader = DataLoader(train_set, batch_size=batch_size, pin_memory=True, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, pin_memory=True, shuffle=True)

    return train_loader, test_loader


def load_CMap(path):

    if ".gct" in path:
        gct = parse(path)
        df = gct.data_df
    elif ".csv" in path:
        df = pd.read_csv(path, sep = ',')
    elif ".txt" in path:
        df = pd.read_csv(path, sep='\t')
    else:
        print("Unsupported file type - options are gctx, gctx, txt, and csv")
        return -1

    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns = ['Unnamed: 0'])
        print('Unnamed dropped')

    return df

def vectorize(df: pd.DataFrame, scale_min_max=True, scale_log2_1=True):
    vals = df.values.transpose()
    keys = np.array(df.keys())

    if scale_log2_1:
        vals = log2_1(vals)

    if scale_min_max:
        vals = min_max(vals.T).T

    return list(zip(keys, vals))


def shuffle(vectors):
    shuffle_idx = np.random.permutation(np.arange(len(vectors)))
    return [vectors[i] for i in shuffle_idx]


def min_max(x):
    return (x - np.min(x, 0)) / (np.max(x, 0) - np.min(x, 0))


def log2_1(x):
    return np.log2(x + 1)


class TorchVectors(Dataset):

    def __init__(self, train_pairs):
        self.train_pairs = train_pairs

    def __len__(self):
        return len(self.train_pairs)

    def __getitem__(self, idx):
        sample = self.train_pairs[idx][1]
        return torch.from_numpy(sample).float()
