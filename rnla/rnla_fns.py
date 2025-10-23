from pathlib import Path

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity


def rsvd(A: np.array, rank: int):
    Omega = np.random.normal(size=(A.shape[-1], rank))
    Y = A @ Omega
    Q, _ = np.linalg.qr(Y)
    C = Q.T @ A
    U, Sigma, Vh = np.linalg.svd(C)
    return Q @ U, Sigma, Vh


def construct_grid(X: np.array, n_components: int, grid_args):
    lower, upper, n_points = grid_args
    X_proj = reduce_dim(X.reshape(-1, X.shape[-1]), n_components=n_components)
    logp_x = fit_dist(X_proj)
    grid = create_grid(lower, upper, n_points)
    z = logp_x(grid.reshape(-1, grid.shape[-1]))
    z = z.reshape(n_points, n_points)
    grid = np.concat([grid, z[..., None]], axis=-1)
    return grid


def create_grid(lower: int, upper: int, n_points: int):
    x = np.linspace(lower, upper, n_points)
    y = np.linspace(lower, upper, n_points)
    xx, yy = np.meshgrid(x, y)
    grid = np.stack([xx, yy], axis=-1)  # [n_points,n_points,2]
    return grid


def reduce_dim(X: np.array, n_components):
    # X: [B,D]
    assert X.ndim == 2, f"{X.ndim == 2}"
    pca = PCA(n_components=n_components)
    X_proj = pca.fit_transform(X)  # [B,n_components]
    return X_proj


def fit_dist(X: np.array):
    # X: [B,D]
    kde = KernelDensity(kernel="gaussian", bandwidth=2.0).fit(X)
    return kde.score_samples


def plot_contour(grid, save_path=None):
    sns.set(font_scale=4)
    sns.set_palette("Set2")
    plt.figure(figsize=(12, 10), dpi=100)
    cp = plt.contourf(
        grid[..., 0],
        grid[..., 1],
        grid[..., -1],
        levels=50,
        cmap="viridis",
    )
    plt.colorbar(cp, label=r"$\log(p(x))$")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.axis("equal")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(Path(save_path) / "contour.pdf")
        plt.savefig(Path(save_path) / "contour.png")
    plt.show()
