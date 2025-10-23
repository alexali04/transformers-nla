import itertools
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler


def get_loaders(args):
    train_loaderS, test_loaderS, max_seq = [], [], 0
    for data_path in args.data_path:
        loaders = construct_loader(data_path, args.bsz)
        train_loaderS.append(iter(loaders["train"]))
        test_loaderS.append(iter(loaders["test"]))
        max_seq = max(max_seq, loaders["n_seq"])
    setattr(args, "n_seq", max_seq)
    setattr(args, "op", loaders["op"])
    return train_loaderS, test_loaderS


def construct_loader(datapath, bsz):
    loaders = {}
    for split in ["train", "test", "val"]:
        dataset = MemoryMap(datapath, split=split)
        loaders[split] = DataLoader(dataset, batch_size=bsz, sampler=InfiniteSampler(dataset))
    loaders["n_seq"] = dataset.n_seq
    loaders["op"] = dataset.op
    return loaders


class InfiniteSampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return itertools.cycle(range(len(self.data_source)))

    def __len__(self):
        return len(self.data_source)


class MemoryMap(Dataset):
    def __init__(self, datapath, split):
        datapath = Path(datapath)
        self.dtype, dtype = torch.float32, np.float32
        info = pickle.load(open(datapath / "info.pkl", mode="rb"))
        shapes = info[split]
        self.X = np.memmap(datapath / f"X_{split}.dat", mode="r", dtype=dtype, shape=shapes["X"])
        self.Y = np.memmap(datapath / f"Y_{split}.dat", mode="r", dtype=dtype, shape=shapes["Y"])
        self.n_seq = self.X.shape[-2]
        self.op = info["kwargs"]["op"]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        x = torch.tensor(self.X[index], dtype=self.dtype)
        y = torch.tensor(self.Y[index], dtype=self.dtype)
        return x, y
