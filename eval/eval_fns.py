import re

import numpy as np
import torch
import wandb

from data.create_fns import OPS
from eval.eval_utils import MATRIX_LIST, construct_matrix
from loss.loss_fns import MATR_LOSS, SCALAR_LOSS, pre_process
from nn.nn_fns import forward_pass


def print_stats(args, x, y_hat, y, loss, idx, lr, rel_errs, grad_norm):
    sl = (slice(None, 2),) + (slice(None, 3),) * (y.ndim - 1) if y.ndim > 2 else (slice(None, 10), 0)
    regular_stats = {"loss": loss, "iter": idx, "lr": lr, "grad_norm": grad_norm}

    additional_stats = {}
    if y_hat.ndim == 2:
        additional_stats["target_var"] = torch.mean(torch.var(y, dim=0)).item()
        additional_stats["pred_var"] = torch.mean(torch.var(y_hat, dim=0)).item()
        var_str = f"Tar var: {additional_stats['target_var']:.3e} | Pred var: {additional_stats['pred_var']:.3e}"

    elif y_hat.ndim == 3:
        additional_stats["output_var"] = torch.var(y_hat, dim=0).mean().item()
        # additional_stats["cond_num"] = torch.linalg.cond(x, p=2).mean()
        # var_str = f"Out var: {additional_stats['output_var']:.3e} | Cond num: {additional_stats['cond_num']:.3e}"
        var_str = f"Out var: {additional_stats['output_var']:.3e}"

    print(f"Size: {x.shape[-1]}x{x.shape[-1]}")
    print(f"Targets:\n{y[sl]} \nPredictions:\n{y_hat[sl]}")
    print(f"{args.loss} Loss: {loss:.3e} | Iter: {idx:,d} | LR: {lr:.3e} | GN: {grad_norm:.3e}")
    print(var_str)

    for norm, rel_err in rel_errs.items():
        regular_stats[f"rel_err_{norm}"] = rel_err.item()
        print(f"Rel error {norm}: {rel_err:.3e}")
    print("\n")

    return regular_stats, additional_stats


def compute_safe_rel_errs(y, y_hat, ord, xnp=torch):
    assert y.shape == y_hat.shape
    axis = tuple(range(1, y.ndim))
    norm = xnp.linalg.norm(y, axis=axis, keepdims=True, ord=ord)
    norm = xnp.where(norm > 1e-7, norm, xnp.ones_like(norm))
    rel_err = xnp.linalg.norm(y - y_hat, axis=axis, keepdims=True, ord=ord) / norm
    return rel_err


def compute_rel_errs(y, y_hat):
    if y_hat.ndim == 3:
        rel_err_keys = ["fro", "nuc", 1, 2]
    elif y_hat.ndim == 2:
        rel_err_keys = [1]

    rel_errs = {}
    for norm_key in rel_err_keys:
        rel_err = compute_safe_rel_errs(y=y, y_hat=y_hat, ord=norm_key)
        rel_errs[f"{norm_key}"] = rel_err.mean()

    return rel_errs


def eval_model(model, test_loaderS, args, R, device, dtype):
    test_rel_err_means, losses = {}, {}

    for test_loader in test_loaderS:
        batch = next(test_loader)
        x, y = batch[0].to(device, dtype), batch[1].to(device, dtype)
        y_hat = forward_pass(args, model, x, R)
        y, y_hat = pre_process(y, y_hat, x, args.loss)
        sl = (slice(None, 2),) + (slice(None, 3),) * (y.ndim - 1) if y.ndim > 2 else (slice(None, 10), 0)
        print(f"Test Set on {args.op} | Size: {x.shape[-1]}x{x.shape[-1]}")
        print(f"Targets:\n {y[sl].cpu().detach().numpy()}")
        print(f"Predictions:\n {y_hat[sl].cpu().detach().numpy()}")
        print("\n")

        norms = [1, 2, "fro", "nuc"] if y.ndim == 3 else [1]
        for norm in norms:
            test_rel_errors = compute_safe_rel_errs(y=y, y_hat=y_hat, ord=norm)
            test_rel_err_means[f"{x.shape[-1]}_{norm}"] = test_rel_errors.mean()

        LOSS_FNS = MATR_LOSS if y.ndim == 3 else SCALAR_LOSS
        for loss_name in LOSS_FNS.keys():
            loss = LOSS_FNS[loss_name](y_hat, y)
            print(f"{loss_name} Loss: {loss:.3e}")
            losses[f"{x.shape[-1]}_{loss_name}"] = loss.item()

        print(f"Rel error: {test_rel_errors.mean():.3e}\n")

    return test_rel_err_means, losses


def eval_conv_model(model, device, dtype, args, matrix_type):
    op = OPS[args.op]

    sizes = args.eval_conv_sizes
    bsz = args.eval_conv_bsz

    for size in sizes:
        matrix = construct_matrix(
            n_dim=size, args=args, matrix_type=matrix_type, bsz=bsz, psd=False, add_rhs=False, symmetric=True
        )
        matrix = matrix.to(device, dtype)
        y = op(matrix.cpu().detach().numpy())
        y_hat = forward_pass(args, model, matrix, R=None).cpu().detach().numpy()

        rel_err = compute_safe_rel_errs(y=y, y_hat=y_hat, ord=1, xnp=np)

        print(f"{matrix_type} | {size}x{size} | Rel err: {rel_err.mean():.3e}")


def exists(args, key):
    if hasattr(args, key):
        if getattr(args, key) is not None:
            return True
    return False


def eval_OOD_matrices(model, args, R, device, dtype):
    op = OPS[args.op]
    psd = True if args.op in ["log_det", "inv", "solve"] else False
    add_rhs = True if args.op in ["solve", "lstsq"] else False
    is_lstsq = True if args.op in ["lstsq"] else False
    eval_sizes = args.eval_sizes if exists(args, "eval_sizes") else [args.n_seq]
    eval_bsz = args.bsz if not hasattr(args, "eval_bsz") else args.eval_bsz
    eval_ma_rel_errs = {}
    eval_ma_losses = {}

    with torch.no_grad():
        for eval_size in eval_sizes:
            for matrix_type in MATRIX_LIST:
                matrix = construct_matrix(
                    eval_size,
                    args,
                    matrix_type,
                    bsz=eval_bsz,
                    psd=psd,
                    add_rhs=add_rhs,
                    is_lstsq=is_lstsq,
                    symmetric=False,
                )
                matrix = matrix.to(device, dtype)
                y_hat = forward_pass(args, model, matrix, R=R)
                y = op(matrix.cpu().detach().numpy())
                y, y_hat = pre_process(torch.tensor(y.copy(), device=device), y_hat, matrix, args.loss)

                sl = (slice(None, 2),) + (slice(None, 3),) * (y.ndim - 1) if y.ndim > 2 else (slice(None, 3), 0)
                print(f"Matrix type: {matrix_type} | Size: {eval_size}x{eval_size} | R: {args.n_layer}")
                print(f"Targets:\n {y[sl]}")
                print(f"Predictions:\n {y_hat[sl]}\n")

                norms = [1, 2, "fro", "nuc"] if y.ndim == 3 else [None]
                LOSS_FNS = MATR_LOSS if y.ndim == 3 else SCALAR_LOSS
                for norm_key in norms:
                    rel_err = compute_safe_rel_errs(y=y, y_hat=y_hat, ord=norm_key, xnp=torch).cpu().numpy()
                    text = f"_{norm_key}" if norm_key is not None else ""
                    eval_ma_rel_errs[f"{matrix_type}_rel_err_{eval_size}{text}"] = np.mean(rel_err)

                for loss_name in LOSS_FNS.keys():
                    loss = LOSS_FNS[loss_name](y_hat, y)
                    text = f"_{loss_name}"
                    eval_ma_losses[f"{matrix_type}_loss_{eval_size}{text}"] = loss.item()

    print(f"Matrix Size {eval_size}x{eval_size} | Batch size {eval_bsz}")
    for key, value in eval_ma_losses.items():
        print(f"{key}{' ' * (15 - len(key))}: {value:.3e}")

    print("\n")
    for key, value in eval_ma_rel_errs.items():
        print(f"{key}{' ' * (15 - len(key))}: {value:.3e}")

    return eval_ma_rel_errs, eval_ma_losses


def generate_info(args, Rs):
    info = {}
    normS = [1, 2, "fro", "nuc"] if args.op in ["inv"] else []
    for ma in MATRIX_LIST:
        inner = {ma: {"R": np.array(Rs)}}
        for norm in normS:
            inner[ma][f"rel_err_{norm}"] = np.zeros(len(Rs))
        info.update(inner)
    return info


def recurrent_eval(model, args, Rs, device, dtype):
    print("-" * 70)
    info = {ma: {"R": np.array(Rs), "rel_err": np.zeros(len(Rs))} for ma in MATRIX_LIST}
    for rdx, R in enumerate(Rs):
        eval_ma, _ = eval_OOD_matrices(model, args, R=R, device=device, dtype=dtype)
        for key, rel_err in eval_ma.items():
            ma = re.match(r"^([^_]*)", key).group(1)
            spacing = 18 - len(ma)
            if args.op in ["inv"]:
                norm = key.split("_")[-1]
                if norm == "nuc":
                    info[ma]["rel_err"][rdx] = rel_err
                    print(f"{ma} {R=} {' ' * spacing}Rel Err: {rel_err:2e}")
            else:
                info[ma]["rel_err"][rdx] = rel_err
                print(f"{ma} {R=} {' ' * spacing}Rel Err: {rel_err:2e}")

    quants = get_quantiles(info)
    for rdx, R in enumerate(Rs):
        q1, q2, q3 = quants[rdx]
        print(f"R:{R:3,d} | Q25: {q1:.3e} | Q50: {q2:.3e} | Q75: {q3:.3e}")

    return info


def get_quantiles(info):
    matrix_types = list(info.keys())
    Rs = info[matrix_types[0]]["R"]
    qs = [0.25, 0.5, 0.75]
    quants = np.zeros((len(Rs), len(qs)))
    for rdx in range(len(Rs)):
        data = np.zeros(len(matrix_types))
        for mdx, (ma, val) in enumerate(info.items()):
            data[mdx] = val["rel_err"][rdx]
        quants[rdx] = np.quantile(data, qs)
    return quants


def eval_recurrent_convolution(model, args, device):
    eval_dims = [512, 1024, 2048]
    for eval_dim in eval_dims:
        x = np.random.randn(5, eval_dim, eval_dim)
        x = x @ np.swapaxes(x, -1, -2)

        x = torch.from_numpy(x).float().to(device)
        if args.operation == "log_det":
            ys = torch.logdet(x)
        elif args.operation == "largest_eigenvalue":
            ys = torch.linalg.eigvalsh(x)[:, -1]

        y_hats = forward_pass(args, model, x, R=None)
        y_hats, ys = y_hats.cpu().detach().numpy(), ys.cpu().detach().numpy()

        rel_error = np.abs(y_hats - ys) / np.abs(ys)
        if args.wandb:
            wandb.summary[f"generalization_error_{eval_dim}"] = rel_error.mean()
        else:
            print(f"Rel error for {eval_dim}x{eval_dim}: {rel_error.mean()}")
