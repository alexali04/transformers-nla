import time

import numpy as np
from cola.linalg.inverse.cg import run_batched_cg
from cola.linalg.inverse.gmres import gmres as _gmres


def run_krylov(alg, alg_kwargs, x0, soln, max_iters):
    alg_fn = ALGS[alg]
    res = np.zeros(max_iters)
    times = np.zeros(max_iters)
    print(f"\n{alg}")
    for tdx in range(max_iters):
        t0 = time.time()
        approx = alg_fn(x0=x0, max_iters=tdx, **alg_kwargs)
        t1 = time.time()
        res[tdx] = np.linalg.norm(soln - approx)
        times[tdx] = t1 - t0
        print(f"{tdx:,d} | {res[tdx]:.3e}")
    return res, times


def cg(A, b, x0, max_iters, tol, preconditioner, pbar):
    soln, *_ = run_batched_cg(A, b, x0, max_iters, tol, preconditioner, pbar)
    approx = A @ soln
    return approx.cpu().numpy()


def gmres(A, b, x0, max_iters, tol, preconditioner, pbar):
    if max_iters > 0:
        soln, *_ = _gmres(A, b, x0=x0, max_iters=max_iters, tol=tol, P=preconditioner, use_triangular=True, pbar=pbar)
        approx = A @ soln
    else:
        approx = A @ x0
    return approx.cpu().numpy()


ALGS = {"CG": cg, "GMRES": gmres}
