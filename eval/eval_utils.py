import gzip
import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import wandb
import wget
from scipy.io import mmread
from scipy.linalg import toeplitz

MATRIX_LIST = [
    "Identity",
    "Diagonal",
    "Zero",
    "Low Rank",
    "Spectral Ratio",
    "Toeplitz",
    "Kronecker",
    "Gaussian",
    "MM Slice",
    "MM Dim",
]


MM_FILES = {
    "Harwell-Boeing/bcsstruc1": [  # https://math.nist.gov/MatrixMarket/data/Harwell-Boeing/bcsstruc1/bcsstruc1.html
        "bcsstk01",
        "bcsstk02",
        "bcsstk03",
        "bcsstk04",
        "bcsstk05",
        "bcsstk06",
        "bcsstk07",
        "bcsstk08",
        "bcsstk09",
        "bcsstk10",
        "bcsstk11",
        "bcsstk12",
        "bcsstk13",
        "bcsstm01",
        "bcsstm02",
        "bcsstm03",
        "bcsstm04",
        "bcsstm05",
        "bcsstm06",
        "bcsstm07",
        "bcsstm08",
        "bcsstm09",
        "bcsstm10",
        "bcsstm11",
        "bcsstm12",
        "bcsstm13",
    ],
    "Harwell-Boeing/bcsstruc2": [  # https://math.nist.gov/MatrixMarket/data/Harwell-Boeing/bcsstruc2/bcsstruc2.html
        "bcsstk14",
        "bcsstk15",
        "bcsstk16",
        "bcsstk17",
        "bcsstk18",
    ],
    "Harwell-Boeing/bcsstruc3": [  # https://math.nist.gov/MatrixMarket/data/Harwell-Boeing/bcsstruc3/bcsstruc3.html
        "bcsstk19",
        "bcsstk20",
        "bcsstk21",
        "bcsstk22",
        "bcsstk23",
        "bcsstk24",
        "bcsstk25",
        "bcsstm19",
        "bcsstm20",
        "bcsstm21",
        "bcsstm22",
        "bcsstm23",
        "bcsstm24",
        "bcsstm25",
    ],
    "Harwell-Boeing/bcsstruc4": [  # https://math.nist.gov/MatrixMarket/data/Harwell-Boeing/bcsstruc4/bcsstruc4.html
        "bcsstk26",
        "bcsstk27",
        "bcsstk28",
        "bcsstm26",
        "bcsstm27",
    ],
    "misc/cylshell": [  # https://math.nist.gov/MatrixMarket/data/misc/cylshell/cylshell.html
        "s1rmq4m1",
        "s2rmq4m1",
        "s3rmq4m1",
        "s1rmt3m1",
        "s2rmt3m1",
        "s3rmt3m1",
        "s3dkq4m2",
        "s3dkt3m2",
        "s3rmt3m3",
    ],
    "Harwell-Boeing/lanpro": [
        "nos1",
        "nos2",
        "nos3",
        "nos4",
        "nos5",
        "nos6",
        "nos7",
    ],
    "Harwell-Boeing/laplace": [
        "gr_30_30",
    ],
    "Harwell-Boeing/psadmit": ["662_bus", "494_bus", "685_bus", "1138_bus"],
}

MM_PATH = "https://math.nist.gov/pub/MatrixMarket2/"


def log_into_wandb(args, model_args):
    log_args = args.__dict__
    log_args["model"] = model_args.model
    log_args["n_seq"] = model_args.n_seq
    log_args["operation"] = model_args.op
    log_args["n_head"] = model_args.n_head
    log_args["n_layer"] = model_args.n_layer
    log_args["n_embd"] = model_args.n_embd
    log_args["pos_enc"] = model_args.pos_enc
    log_args["dim_proportion"] = model_args.dim_proportion
    wandb.config.update(log_args)


def get_max_eigval(matrix):
    evs = np.linalg.eigvals(matrix)  # absolute in magnitude or value?
    return np.max(np.abs(evs), axis=-1)


def find_factors(n):
    factors = set()
    for i in range(1, int(np.sqrt(n) + 1)):
        if n % i == 0:
            factors.add(tuple(sorted((i, n // i))))
    return list(list(x) for x in factors)


def log_results(y_hat, y, task):
    print(f"Predicted {task}:\n {y_hat}")
    print(f"True {task}:\n {y}\n")


def construct_matrix(
    n_dim,
    args,
    matrix_type: str,
    bsz: int,
    psd: bool,
    add_rhs: bool,
    symmetric: bool,
    is_lstsq: bool,
):
    eps = 1e-4
    if matrix_type == "Identity":
        X = torch.eye(n_dim)[None].broadcast_to(bsz, n_dim, n_dim)

    elif matrix_type == "Diagonal":
        X = torch.diag_embed(torch.randn(bsz, n_dim))

    elif matrix_type == "Permutation":
        X = torch.stack([torch.eye(n_dim)[torch.randperm(n_dim)] for _ in range(bsz)])

    elif matrix_type == "Zero":
        X = torch.zeros(1, n_dim, n_dim)
        X += eps * torch.eye(n_dim) if args.op == "inv" else 0

    elif matrix_type == "Low Rank":
        L = torch.randn(bsz, n_dim, int(np.sqrt(n_dim)))
        X = L @ L.transpose(-1, -2)
        X[..., torch.arange(n_dim), torch.arange(n_dim)] += eps

    elif matrix_type == "Spectral Ratio":
        A = np.random.randn(bsz, n_dim, n_dim)
        Q, _ = np.linalg.qr(A)
        diags = np.random.rand(bsz, n_dim - 2) * 10 + 1.5  # sample [1.5, 11.5]
        second = np.random.rand(bsz, 1) * 11.5 + 11.5
        first = np.sqrt(2) * second

        diags = np.concatenate([diags, second, first], axis=-1)
        D = np.apply_along_axis(np.diag, -1, diags)
        X = Q @ D @ np.swapaxes(Q, -1, -2)

        X = (X + np.swapaxes(X, -1, -2)) / 2  # symmetric
        X = torch.tensor(X, dtype=torch.float32)

    elif matrix_type == "Toeplitz":
        matrices = []
        for _ in range(bsz):
            cols = np.abs(np.random.randn(n_dim))
            T = toeplitz(cols)
            matrices.append(torch.from_numpy(T).float())
        X = torch.stack(matrices)

    elif matrix_type == "Kronecker":
        n_sqrt = int(np.ceil(np.sqrt(n_dim)))
        matrices = []
        A = torch.randn(bsz, n_sqrt, n_sqrt)
        B = torch.randn(bsz, n_sqrt, n_sqrt)
        matrices = []
        for bdx in range(bsz):
            X = torch.kron(A[bdx] + A[bdx].T, B[bdx] + B[bdx].T)
            matrices.append(X)
        X = torch.stack(matrices)
        X = X[..., :n_dim, :n_dim]

    elif matrix_type == "Gaussian":
        X = torch.randn(bsz, n_dim, n_dim)

    elif matrix_type == "MM Slice":
        X = read_mm_matrix(n_dim=n_dim, bsz=bsz, process_fn=slice_them)
        X[..., torch.arange(n_dim), torch.arange(n_dim)] += eps

    elif matrix_type == "MM Dim":
        X = read_mm_matrix(n_dim=n_dim, bsz=bsz, process_fn=project_them)
        X[..., torch.arange(n_dim), torch.arange(n_dim)] += eps

    else:
        raise ValueError(f"Invalid matrix type: {matrix_type}")

    if psd and (matrix_type not in ["MM Slice", "MM Dim", "Low Rank"]):
        X = X.cpu().numpy()
        X = psdify(X, eps=eps)
        X = torch.from_numpy(X)

    if add_rhs:
        rhs = torch.randn(size=(X.shape[:-1] + (1,)))
        rhs = rhs / torch.linalg.norm(rhs, dim=(1, 2), keepdim=True)
        if is_lstsq:
            X = X[..., :5]
        X = torch.cat([X, rhs], dim=-1)

    if symmetric:
        X = torch.tensor(symmetrize(X.cpu().numpy(), eps=eps))

    return X


def symmetrize(X: np.array, eps: float):
    # X: [B,N,N]
    Xup = np.tril(X, k=-1).swapaxes(-1, -2)
    Xlow = np.tril(X)
    X = Xup + Xlow
    X[..., np.arange(X.shape[-1]), np.arange(X.shape[-1])] += eps
    # X = (X + np.swapaxes(X, -1, -2)) / 2 + eps * Id
    return X


def psdify(X: np.array, eps: float):
    X = X @ X.swapaxes(-1, -2)
    X += eps * np.eye(X.shape[-1])
    return X


def get_all_mm_matrices(fpath=Path("datasets/matrix_market")):
    fpath_mm_s = list(fpath.rglob("*.npy"))
    random.shuffle(fpath_mm_s)

    As = []
    for idx, fpath_mm in enumerate(fpath_mm_s):
        print(f"{idx:3,d} | {fpath_mm.stem}")
        A = np.load(fpath_mm)
        # A = process_fn(A=A, n_dim=n_dim)
        diff = np.linalg.norm(A - A.T)
        assert diff < 1e-6, f"{fpath_mm} is not symmetric"
        As.append((fpath_mm.stem, A))

    return As


def read_mm_matrix(n_dim, bsz, process_fn, fpath=Path("datasets/matrix_market")):
    fpath_mm_s = list(fpath.rglob("*.npy"))
    random.shuffle(fpath_mm_s)
    bsz = min(bsz, len(fpath_mm_s))
    As = np.zeros(shape=(bsz, n_dim, n_dim))

    for idx, fpath_mm in enumerate(fpath_mm_s[:bsz]):
        A = np.load(fpath_mm)
        A = process_fn(A=A, n_dim=n_dim)

        if A.shape[0] != n_dim:
            print(f"Matrix {fpath_mm} has shape {A.shape}")
            print(f"Skipping {fpath_mm} because it has shape {A.shape}")
            continue

        diff = np.linalg.norm(A - A.T)
        assert diff < 1e-6, f"{fpath_mm} is not symmetric"
        As[idx] = A

    return torch.from_numpy(As).float()


def slice_them(A, n_dim):
    out = A[:n_dim, :n_dim]
    out = out / np.max(np.abs(out))
    return out


def project_them(A, n_dim):
    v = np.random.randn(n_dim, A.shape[0])
    out = v @ A @ v.T
    out = out / np.max(np.abs(out))
    return out


def download_mm_matrices(fpath=Path("datasets/matrix_market"), max_size=int(5e3)):
    os.makedirs(fpath, exist_ok=True)
    for path in MM_FILES.keys():
        search_path = f"{MM_PATH}{path}"
        print(f"Searching {search_path}")
        for ending in MM_FILES[path]:
            url = f"{MM_PATH}{path}/{ending}.mtx.gz"
            print(f"Downloading {url}")
            wget.download(url, out=fpath.as_posix())
            with gzip.open(fpath / f"{ending}.mtx.gz", "rb") as f_in:
                with open(fpath / f"{ending}.mtx", "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            os.remove(fpath / f"{ending}.mtx.gz")
            A = mmread(fpath / f"{ending}.mtx")
            print(f"Matrix: {ending} has shape {A.shape=}")
            if A.shape[0] <= max_size:
                A = np.array(A.todense())
                np.save(fpath / f"{ending}.npy", A)
                print(f"Saved: {fpath / f'{ending}.npy'}")


def is_invertible(matrix):
    return matrix.shape[0] == matrix.shape[1] and np.linalg.matrix_rank(matrix) == matrix.shape[0]
