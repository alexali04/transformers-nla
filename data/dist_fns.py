import time
from functools import partial

import numpy as np
import torch

from eval.eval_utils import read_mm_matrix, slice_them


def factorize(N: int):
    # finds triplet factors of N, a * b * g = N, d * e * f = N, r = whatever
    factors = []
    for a in range(1, N + 1):
        if N % a == 0:
            for b in range(1, N // a + 1):
                if (N // a) % b == 0:
                    g = N // (a * b)
                    if a * b * g == N:
                        factors.append([a, b, g])
    return factors


def construct_einsum_matrix(vec, bsz):
    na, nb, ng, nd, ne, nf, nr = vec
    X = np.eye(N=na * nb * ng).reshape(-1, na, nb, ng)[None].repeat(bsz, axis=0)
    A = np.random.normal(size=(bsz, na, ng, nd, nf, nr))
    B = np.random.normal(size=(bsz, nb, ng, ne, nf, nr))
    # A = np.random.laplace(size=(bsz, na, ng, nd, nf, nr))
    # B = np.random.uniform(size=(bsz, nb, ng, ne, nf, nr))
    mA = np.random.binomial(n=1, p=0.8, size=(bsz, na, ng, nd, nf, nr))
    mB = np.random.binomial(n=1, p=0.8, size=(bsz, nb, ng, ne, nf, nr))
    A = mA * A
    B = mB * B
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X = torch.tensor(X, device=device)
    A = torch.tensor(A, device=device)
    B = torch.tensor(B, device=device)
    Y = torch.einsum("BZabg,Bagdfr,Bbgefr->BZdef", X, A, B)
    Y = Y.reshape(X.shape[0], X.shape[1], -1)
    return Y.cpu().numpy()


def einsum(B: int, N: int, struct_n: int, **_):
    struct_n = B if struct_n > B else struct_n
    block_n = B // struct_n
    all_factors = factorize(N)
    indices = np.arange(len(all_factors))
    Y = np.zeros((B, N, N))

    def sample():
        na, nb, ng = all_factors[np.random.choice(indices)]
        nd, ne, nf = all_factors[np.random.choice(indices)]
        nr = int(np.random.randint(low=1, high=min(na, ne) + 1, size=1))
        vec = (na, nb, ng, nd, ne, nf, nr)
        return vec

    for i in range(struct_n):
        print(f"Processing {i + 1}th factorization")
        tic = time.time()
        Y[i * block_n : (i + 1) * block_n] = construct_einsum_matrix(sample(), bsz=block_n)
        toc = time.time()
        print(f"Time taken: {toc - tic:.2f} seconds")

    if B - block_n * struct_n > 0:
        print(f"Processing {struct_n + 1}th factorization")
        tic = time.time()
        Y[struct_n * block_n :, :] = construct_einsum_matrix(sample(), bsz=B - block_n * struct_n)
        toc = time.time()
        print(f"Time taken: {toc - tic:.2f} seconds")

    return Y


def diag_decay(B: int, N: int, diag_fn, alpha_fn, **_):
    xgrid = np.linspace(0, 1, num=N)[None]
    scale = np.random.uniform(1, 3, size=(B, 1))
    alpha = alpha_fn(B=B)
    diags = diag_fn(xgrid, alpha)
    diags = scale * diags

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    diags = torch.tensor(diags, dtype=dtype, device=device)
    # Q = torch.eye(N, device=device, dtype=dtype)[None].broadcast_to(B, N, N)
    Q = torch.randn(B, N, N, dtype=dtype, device=device)
    if N <= 100:
        Q, _ = torch.linalg.qr(Q)
    D = diags[..., None] * Q.transpose(-1, -2)
    X = Q @ D

    X = X.cpu().numpy()

    return X


def rbf(B: int, N: int, **_):
    x = np.random.uniform(low=0, high=10, size=N)
    # x = np.linspace(0, 1, num=N)
    ls = np.random.uniform(low=0.1, high=10, size=(B, 1, 1))
    sigma = np.random.uniform(low=0.1, high=1, size=(B, 1, 1))

    ker = x[None, :] - x[:, None]
    ker = np.exp(-((ker[None] / ls) ** 2.0)) + sigma * np.eye(N)[None]
    return ker


def gaussian(B: int, N: int, **_):
    X = np.random.normal(size=(B, N, N))
    return X


def scale_singular_values(A, l_max=2, l_min=1):
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    S *= (l_max - l_min) / (S[:, 0][:, None] - S[:, -1][:, None])
    S += l_min - S[:, -1][:, None]
    return U @ torch.diag_embed(S) @ Vh


def gaussian_lstsq(B: int, N: int, **_):
    X = np.random.normal(size=(B, N, 5))
    X = torch.tensor(X, dtype=torch.float64)
    X = scale_singular_values(X, l_max=5, l_min=1)
    X = X.cpu().numpy()
    rhs = np.random.normal(size=(B, N, 1))
    rhs = rhs / np.linalg.norm(rhs, axis=(1, 2), keepdims=True)
    X = np.concat([X, rhs], axis=-1)
    return X


def laplace(B: int, N: int, **_):
    X = np.random.laplace(size=(B, N, N))
    return X


def unif(B: int, N: int, **_):
    X = np.random.uniform(low=-1.0, high=1.0, size=(B, N, N))
    return X


def normal_sq(B: int, N: int, **_):
    X = np.random.randn(B, N, N)
    out = X @ np.swapaxes(X, -1, -2)
    return out


def lin(B: int, N: int, **_):
    # X = np.broadcast_to(np.linspace(0, 1, num=N)[None], shape=(B, N))
    X = np.random.uniform(low=0, high=10, size=(B, N))
    return X


def ber(B: int, N: int, prob=0.75, **_):
    X = np.random.binomial(n=1, p=prob, size=(B, N, N))
    return X


def poly_decay(xgrid, alpha):
    diag = 1 - xgrid**alpha
    diag[..., -1] += 1e-4
    return diag


def poly_alpha(B: int):
    alpha = np.random.uniform(0, 5, size=(B, 1))
    return alpha


def cos_decay(xgrid, alpha):
    diag = np.sort(0.5 * (1 + np.cos(np.pi * alpha * xgrid)))[..., ::-1]
    return diag


def cos_alpha(B: int):
    bsz = int(B * 0.2)
    a1 = np.random.uniform(0.05, 0.1, size=(bsz, 1))
    a2 = np.random.uniform(1.0, 4.0, size=(bsz, 1))
    alpha = np.random.uniform(0.3, 0.9, size=(B - 2 * bsz, 1))
    alpha = np.concat((a1, alpha, a2), axis=0)
    np.random.shuffle(alpha)
    return alpha


def inv_decay(xgrid, alpha):
    loc = 0.5
    diag = 1 / (1 + np.exp(alpha * (xgrid - loc)))
    return diag


def inv_alpha(B: int):
    alpha = np.random.uniform(0.3, 30.0, size=(B, 1))
    return alpha


def log_decay(xgrid, alpha):
    diag = 1 - np.log(1 + alpha * xgrid) / np.log(1 + alpha)
    diag[..., -1] += 1e-4
    return diag


def log_alpha(B: int):
    alpha = np.random.uniform(1.0, 1e3, size=(B, 1))
    return alpha


def linear_decay(xgrid, alpha):
    diag = -alpha * xgrid + 1.0
    corr = 1e-4 * np.abs(np.random.normal(size=diag.shape))
    corr = np.sort(corr)[..., ::-1]
    diag = np.where(diag <= 0.0, corr, diag)
    return diag


def linear_alpha(B: int):
    alpha = np.random.uniform(low=1e0, high=1e2, size=(B, 1))
    return alpha


DECAYS = {
    "poly": poly_decay,
    "cos": cos_decay,
    "inv": inv_decay,
    "log": log_decay,
    "linear": linear_decay,
}


def mm_slice(B: int, N: int, **_):
    X = read_mm_matrix(n_dim=N, bsz=B, process_fn=slice_them)
    repeats_n = int(np.ceil(B / X.shape[0]))
    X = np.tile(X, (repeats_n, 1, 1))
    X = X[:B]
    return X


DIST_FNs = {
    "gaussian": gaussian,
    "gaussian_lstsq": gaussian_lstsq,
    "laplace": laplace,
    "unif": unif,
    "einsum": einsum,
    "poly_decay": partial(diag_decay, diag_fn=poly_decay, alpha_fn=poly_alpha),
    "cos_decay": partial(diag_decay, diag_fn=cos_decay, alpha_fn=cos_alpha),
    "inv_decay": partial(diag_decay, diag_fn=inv_decay, alpha_fn=inv_alpha),
    "log_decay": partial(diag_decay, diag_fn=log_decay, alpha_fn=log_alpha),
    "linear_decay": partial(diag_decay, diag_fn=linear_decay, alpha_fn=linear_alpha),
    "mm_slice": mm_slice,
    "rbf": rbf,
    "normal_sq": normal_sq,
}
