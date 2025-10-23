from functools import partial

import torch


def pre_process(y, y_hat, x, loss_name):
    if loss_name.startswith("maid"):
        y_pred = y_hat @ x
        Id = torch.eye(y_hat.shape[-1], device=y_hat.device, dtype=y_hat.dtype)[None]
        Id = Id.repeat(y_hat.shape[0], 1, 1)
        y_hat, y = y_pred, Id

    return y, y_hat


def get_loss(name):
    return LOSS_FNS[name]


def compute_norm(y_hat, y, ord):
    loss = torch.linalg.norm(y_hat - y, ord=ord, dim=tuple(range(1, y.ndim)))
    return torch.mean(loss)


def compute_mse_hp(y_hat, y):
    loss = torch.mean((y_hat - y) ** 2.0)
    return loss


def compute_mse(y_hat, y):
    loss = torch.sum((y_hat - y) ** 2.0, dim=tuple(range(1, y.ndim)))
    return torch.mean(loss)


LOSS_FNS = {
    "mse_hp": compute_mse_hp,
    "mse": compute_mse,
    "l1": partial(compute_norm, ord=1),
    "l2": partial(compute_norm, ord=2),
    "fro": partial(compute_norm, ord="fro"),
    "nuc": partial(compute_norm, ord="nuc"),
    "maid_l1": partial(compute_norm, ord=1),
    "maid_l2": partial(compute_norm, ord=2),
    "maid_fro": partial(compute_norm, ord="fro"),
    "maid_nuc": partial(compute_norm, ord="nuc"),
}

SCALAR_LOSS = {
    "mse": compute_mse,
    "l1": partial(compute_norm, ord=1),
}

MATR_LOSS = {
    "l1": partial(compute_norm, ord=1),
    "l2": partial(compute_norm, ord=2),
    "fro": partial(compute_norm, ord="fro"),
    "nuc": partial(compute_norm, ord="nuc"),
}
