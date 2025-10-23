import logging
import pickle
import time
from functools import partial
from pathlib import Path
from typing import Callable

import numpy as np
import yaml

from data.dist_fns import DIST_FNs
from data.op_fns import OPS
from utils.parsers import get_data_parser


def all(B: int, N: int, variances: list[float], eps: float, **kwargs):
    dists = list(DIST_FNs.keys())
    return choose(B, N, variances, eps, dists, **kwargs)


def choose(
    B: int,
    N: int,
    sample_fn,
    variances: list[float],
    eps: float,
    dists: list[str],
    get_rhs: bool,
    **kwargs,
):
    As = np.zeros(shape=(B, N, N))
    num_dist = len(dists)

    chunk_sz = B // num_dist
    for bdx, dist_name in enumerate(dists):
        out = sample_fn(chunk_sz, N, variances, eps, dist_name, **kwargs)
        As[bdx * chunk_sz : (bdx + 1) * chunk_sz] = out

    remainder = B - num_dist * chunk_sz
    if remainder > 0:
        out = sample_fn(remainder, N, variances, eps, dist_name, **kwargs)
        As[num_dist * chunk_sz :] = out
    np.random.shuffle(As)

    if get_rhs:
        rhs = np.random.normal(size=(B, N, 1))
        rhs = rhs / np.linalg.norm(rhs, axis=(1, 2), keepdims=True)
        As = np.concat([As, rhs], axis=-1)
    return As


def sample_identity(B: int, N: int, variances: list[float], eps: float, dist_name: str, **kwargs):
    var = np.random.choice(variances, size=(B, 1, 1))
    dist_fn = DIST_FNs[dist_name]
    X = dist_fn(B, N, **kwargs) * np.sqrt(var)
    N_min = min(X.shape[1], X.shape[2])
    X[..., np.arange(N_min), np.arange(N_min)] += eps
    return X


def sample_symm(B: int, N: int, variances: list[float], eps: float, dist_name: str, **kwargs):
    var = np.random.choice(variances, size=(B, 1, 1))
    dist_fn = DIST_FNs[dist_name]
    X = dist_fn(B, N, **kwargs) * np.sqrt(var)
    X = symmetrize(X, eps=eps)
    return X


def symmetrize(X: np.array, eps: float):
    # X: [B,N,N]
    Xup = np.tril(X, k=-1).swapaxes(-1, -2)
    Xlow = np.tril(X)
    X = Xup + Xlow
    X[..., np.arange(X.shape[-1]), np.arange(X.shape[-1])] += eps
    # X = (X + np.swapaxes(X, -1, -2)) / 2 + eps * Id
    return X


def sample_psdify(B: int, N: int, variances: list[float], eps: float, dist_name: str, **kwargs):
    var = np.random.choice(variances, size=(B, 1, 1))
    dist_fn = DIST_FNs[dist_name]
    X = dist_fn(B, N, **kwargs) * np.sqrt(var)
    X = psdify(X, eps=eps)
    return X


def psdify(X: np.array, eps: float):
    X = X @ X.swapaxes(-1, -2)
    X += eps * np.eye(X.shape[-1])
    return X


def diag_lanczos(B: int, N: int, **_):
    zeros = np.zeros(B)[:, None]
    outliers = np.array([2.5, 3.0])[None, :]
    outliers = np.broadcast_to(outliers, shape=(B, 2))
    inliers = np.random.uniform(low=0.0, high=2.0, size=(B, N - 3))
    X = np.concatenate((zeros, inliers, outliers), axis=1)
    X = X[:, :, None] * np.eye(N)
    return X


def spectral_gap(B: int, N: int, spectral_gap: float, ub: float = 10.0, lb: float = 1.5, **_):
    A = np.random.rand(B, N, N)
    Q, _ = np.linalg.qr(A)

    diags = np.random.rand(B, N - 2) * (ub - lb) + lb
    second_largest = np.random.rand(B, 1) * ub + ub
    largest = spectral_gap * second_largest

    diags = np.concatenate([diags, second_largest, largest], axis=-1)
    D = np.apply_along_axis(np.diag, -1, diags)
    X = Q @ D @ np.swapaxes(Q, -1, -2)

    print("Gut Checking Spectral Ratio")

    X = (X + np.swapaxes(X, -1, -2)) / 2

    x_sample = X[0]
    print(f"Spectral ratio of X's first matrix: {np.linalg.eigvalsh(x_sample)[-1] / np.linalg.eigvalsh(x_sample)[-2]}")

    return X


def create_numeric(operation: Callable, operation_str: str, sample: Callable, **kwargs):
    logging.info(f"Logging {operation_str}")
    root = Path(f"./datasets/{operation_str}")
    seed = 21
    cases = [(kwargs["B"], "train"), (100, "test"), (100, "val")]
    N, B = kwargs["N"], kwargs["B"]
    kwargs.pop("N")
    kwargs.pop("B")

    dtype = np.float32
    root.mkdir(parents=True, exist_ok=True)
    np.random.seed(seed=seed)
    info = {"kwargs": kwargs}
    shapeX = (N, N + 1) if kwargs["get_rhs"] else (N, N)
    shapeX = (N, 5 + 1) if kwargs["op"] in ["lstsq"] else shapeX
    shapeY = operation(np.eye(*shapeX)[None], **kwargs).shape[1:]

    for B, split in cases:
        X = np.memmap(root / f"X_{split}.dat", dtype=dtype, mode="w+", shape=(B,) + shapeX)
        Y = np.memmap(root / f"Y_{split}.dat", dtype=dtype, mode="w+", shape=(B,) + shapeY)
        print(f"Going over: {split} with {B:,d} data points")
        if B >= 100_000:
            chunk_sz = 100_000
            assert B % chunk_sz == 0, f"{B:,d} is not divisible by {chunk_sz:,d}"
            batches = B // chunk_sz

            print(f"Batching {B:,d} data points by {chunk_sz:,d} for {N}x{N}")

            for i in range(batches):
                tic_sample = time.time()
                X_batch = sample(B=chunk_sz, N=N, **kwargs)
                X[i * chunk_sz : (i + 1) * chunk_sz] = X_batch
                toc_sample = time.time()
                tic_compute = time.time()
                Y_batch = operation(X_batch, **kwargs)
                Y[i * chunk_sz : (i + 1) * chunk_sz] = Y_batch
                toc_compute = time.time()
                text = f"Batch {i} of {batches - 1}: {X_batch.shape} | {Y_batch.shape} | "
                text += f"Sampling Time: {(toc_sample - tic_sample):.2f} seconds | "
                text += f"Compute Time: {(toc_compute - tic_compute):.2f} seconds"
                print(text)

        else:
            X[:] = sample(B=B, N=N, **kwargs)
            Y[:] = operation(X, **kwargs)
        X.flush()
        Y.flush()
        info[split] = {"X": X.shape, "Y": Y.shape}
        text = f"Succesfully created {operation_str} dataset task with {B}"
        text += f" data points on {N}x{N} matrices"
        text += f" from {kwargs['distribution']} distribution"
        print(text)

    pickle.dump(info, open(root / "info.pkl", mode="wb"))
    yaml.dump(info, open(root / "config.yaml", mode="w"))


SAMPLE_FNs = {
    "diag_lanczos": diag_lanczos,
    "einsum": partial(sample_symm, dist_name="einsum"),
    "gaussian": partial(sample_symm, dist_name="gaussian"),
    "laplace": partial(sample_symm, dist_name="laplace"),
    "unif": partial(sample_symm, dist_name="unif"),
    "ber": partial(sample_symm, dist_name="ber"),
    "all": partial(sample_symm, dist_name="all"),
    "all_psd": partial(sample_psdify, dist_name="all"),
    "gaussian_psd": partial(sample_psdify, dist_name="gaussian"),
    "gaussian_identity": partial(sample_identity, dist_name="gaussian"),
    "rbf": partial(sample_symm, dist_name="rbf"),
    "normal_sq": partial(sample_identity, dist_name="normal_sq"),
    "diag_decay": partial(sample_identity, dist_name="diag_decay"),
    "spectral_gap": spectral_gap,
    "choose": partial(choose, sample_fn=sample_identity),
    "choose_symm": partial(choose, sample_fn=sample_symm),
    "choose_psd": partial(choose, sample_fn=sample_psdify),
    "gaussian_lstsq": partial(sample_identity, dist_name="gaussian_lstsq"),
}

if __name__ == "__main__":
    parser = get_data_parser()
    aux = vars(parser.parse_args())
    aux["get_rhs"] = True if aux["op"] in ["solve", "lstsq"] else False

    operation_func = OPS[aux["op"]]
    sample_func = SAMPLE_FNs[aux["distribution"]]
    task = f"{aux['op']}_{aux['distribution']}_{aux['description']}"
    create_numeric(operation_func, task, sample_func, **aux)
