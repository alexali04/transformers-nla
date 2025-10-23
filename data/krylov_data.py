from functools import partial

import numpy as np
import torch

from data.dist_fns import rbf
from eval.eval_utils import project_them, slice_them


def get_data(args, device, dtype):
    seed = args.seed
    N = args.n_seq
    if args.case in CASES.keys():
        case_fn = CASES[args.case]
    else:
        transform = TRANS[args.trans]
        case_fn = partial(get_mm, name=args.case, transform=transform)

    A, rhs = case_fn(N, seed, device, dtype)
    A[torch.arange(N), torch.arange(N)] += args.eps
    soln = torch.linalg.solve(A, rhs)
    return A, rhs, soln


def get_mm(N, seed, device, dtype, name, transform):
    A = np.load(f"datasets/matrix_market/{name}.npy")
    A = transform(A, n_dim=N)
    A = torch.tensor(A, device=device, dtype=dtype)
    rhs = torch.randn(N, 1, device=device, dtype=dtype)
    rhs = rhs / torch.linalg.norm(rhs)
    return A, rhs


TRANS = {"slice": slice_them, "proj": project_them}


def get_rbf(N: int, seed: int, device: str, dtype):
    np.random.seed(seed=seed)
    torch.manual_seed(seed=seed)
    ker = rbf(B=1, N=N)
    A = torch.tensor(ker[0], device=device, dtype=dtype)
    rhs = torch.randn(N, 1, device=device, dtype=dtype)
    rhs = rhs / torch.linalg.norm(rhs)
    return A, rhs


def gauss_psd(N: int, seed: int, device: str, dtype):
    eps = 1e-2
    torch.manual_seed(seed=seed)
    A = torch.randn(N, N, device=device, dtype=dtype)
    A = A @ A.T + eps * torch.eye(N, device=device, dtype=dtype)
    rhs = torch.randn(N, 1, device=device, dtype=dtype)
    rhs = rhs / torch.linalg.norm(rhs)
    return A, rhs


CASES = {
    "rbf": get_rbf,
    "gauss": gauss_psd,
}
