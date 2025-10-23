import pickle
import time
from pathlib import Path

import numpy as np
from utils.timing import print_time_taken

from enc_dec.encoders import FPSymbol

root = Path("./datasets/encoded")
seed = 21
N = 5
cases = [(10_000_000, "train"), (100, "test"), (100, "val")]
# cases = [(100, "train"), (100, "test"), (100, "val")]

dtype = np.long
root.mkdir(parents=True, exist_ok=True)
np.random.seed(seed=seed)
encoder = FPSymbol(max_dim=5, precision=3, max_exp=1)
str2int = {enc: idx for idx, enc in enumerate(encoder.symbols)}

info = {}
tic = time.time()
for B, split in cases:
    print(f"Going over: {split} with {B:,d} data points")
    Z = np.random.randn(B, N, N)
    Z = Z @ np.swapaxes(Z, -1, -2)
    ZY = np.linalg.eigvalsh(Z)
    ZY = ZY[:, [-1]]
    X = np.zeros((B, N * N + 2), dtype=dtype)
    Y = np.zeros((B,), dtype=dtype)
    for bdx in range(B):
        if bdx % int(B * 0.1) == 0:
            print(f"Iter: {bdx:,d}")
        mat = encoder.encode(Z[bdx, :, :])
        mat = [str2int[ent] for ent in mat]
        X[bdx, :] = mat
        aux = encoder.encode_value(ZY[bdx, :])
        Y[bdx] = str2int[aux[0]]
    info[split] = {"X": (B, N * N + 2), "Y": (B,)}

    X_memmap = np.memmap(root / f"X_{split}.dat", dtype=dtype, mode="w+", shape=X.shape)
    X_memmap[:] = X[:]
    X_memmap.flush()

    Y_memmap = np.memmap(root / f"Y_{split}.dat", dtype=dtype, mode="w+", shape=Y.shape)
    Y_memmap[:] = Y[:]
    Y_memmap.flush()

pickle.dump(info, open(root / "info.pkl", mode="wb"))
toc = time.time()
print_time_taken(toc - tic, text="Training time: ")
