import numpy as np
import torch


def max_eigvec(X: np.array, **_):
    Y = np.linalg.eigh(X)[-1]
    return Y[..., -1]


def all_eigvals(X: np.array, **_):
    Y = np.linalg.eigh(X)[0]
    return Y[..., ::-1]


def solve(X: np.array, **_):
    rhs = X[..., [-1]]
    Y = np.linalg.solve(X[..., :-1], rhs)
    return Y[..., 0]


def lstsq(X: np.array, **_):
    dtype = torch.float64
    rhs = torch.tensor(X[..., [-1]], dtype=dtype)
    A = torch.tensor(X[..., :-1], dtype=dtype)
    Y, *_ = torch.linalg.lstsq(A, rhs)
    Y = Y.cpu().numpy()
    return Y[..., 0]


def inv(X: np.array, **_):
    Y = np.linalg.inv(X)
    return Y


def sine(X: np.array, **_):
    eps = np.random.normal(size=X.shape) * 0.05
    Y = np.sin(X)
    return Y + eps


def only_rank(X: np.array, rank: int = 16, **_):
    _, Sigma, _ = np.linalg.svd(X)
    return Sigma[..., :rank]


def svd_rank(X: np.array, rank_pct: float = 0.25, **_):
    rank = int(rank_pct * X.shape[-1])
    U, Sigma, Vh = np.linalg.svd(X)
    Sigma = Sigma[..., None, :]
    Y = (U[..., :, :rank] * Sigma[..., :rank]) @ Vh[..., :rank, :]
    return Y


def trace(X: np.array, **_):
    return np.trace(X, axis1=-2, axis2=-1)[:, None]


def largest_eigval(X: np.array, **_):
    Y = np.linalg.eigvalsh(X)[..., -1]
    return Y[:, None]


def min_eigval(X: np.array, **_):
    Y = np.linalg.eigvalsh(X)[..., 0]
    return Y[:, None]


def largest_eigvec(X: np.array, **_):
    Lam, Q = np.linalg.eigh(X)
    return Q[..., -1]


def log_det(X: np.array, **_):
    sign_tuple = np.linalg.slogdet(X)
    return sign_tuple[1][:, None]


def cond_number(X: np.array, **_):
    cond = np.linalg.cond(X, p=2)
    return cond[:, None]


OPS = {
    "max_eigvec": max_eigvec,
    "svd_rank": svd_rank,
    "only_rank": only_rank,
    "inv": inv,
    "lstsq": lstsq,
    "solve": solve,
    "sine": sine,
    "all_eigvals": all_eigvals,
    "largest_eigenvalue": largest_eigval,
    "largest_eigenvec": largest_eigvec,
    "min_eigenvalue": min_eigval,
    "log_det": log_det,
    "condition_number": cond_number,
    "trace": trace,
}
