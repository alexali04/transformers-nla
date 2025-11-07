import pickle
import time
from pathlib import Path

import numpy as np
import torch
import wandb

from data.data_fns import get_loaders
from eval.eval_fns import eval_model, eval_OOD_matrices
from loss.loss_fns import get_loss, pre_process
from nn.common import get_num_params
from nn.nn_fns import SubspaceModel, forward_pass, get_model
from opt.opt_fns import get_optimizer
from opt.scheduler_fns import get_scheduler
from utils.parsers import get_training_parser
from utils.sampler import get_recurrence_length
from utils.timing import append_timestamp, print_time_taken


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.wandb:
        wandb.init(project=args.wandb_proj)
        wandb.config.update(args.__dict__)

    train_loaderS, test_loaderS = get_loaders(args)
    ds_idxS = [idx for idx in range(len(train_loaderS))]
    net = get_model(args)

    tic = time.time()
    net.to(device)
    num_params = get_num_params(net)
    intrinsic_dim = int(args.dim_proportion * num_params)
    print(f"Compressed parameters: {num_params:,}M: -> {intrinsic_dim:,}K")

    tic = time.time()
    model = SubspaceModel(net, intrinsic_dim, device=device)
    toc = time.time()
    print_time_taken(toc - tic, text="SubspaceModel Init time: ")

    opt = get_optimizer(args.optimizer, [model.subspace_params], lr=args.lr, wd=args.wd, args=args)
    loss_fn = get_loss(args.loss)

    tic = time.time()
    model.to(device)
    model.train()
    R, Rs = get_recurrence_length(args)
    sch = get_scheduler(opt=opt, args=args, model=model, train_loader=train_loaderS, device=device, R=R)
    print(f"Model Type: {args.model} | Iters: {args.iters} | Scheduler: {args.sch_name} | Optimizer: {args.optimizer}")

    for idx in range(args.iters):
        batch = next(train_loaderS[np.random.choice(ds_idxS)])
        x, y = batch[0].to(device), batch[1].to(device)
        R_i = Rs[idx] if Rs is not None else None
        y_out = forward_pass(args, model, x, R_i)
        y_hat, y = pre_process(x=x, y_hat=y_out, y=y, args=args)

        loss = loss_fn(y_hat, y)
        loss.backward()
        opt.step()
        opt.zero_grad()
        sch.step()

        # mean element-wise variance (batch matrix variance)
        out_var_matr = torch.var(y_out, dim=0)
        out_var = out_var_matr.mean().item()

        dim = tuple(range(1, y.ndim))

        rel_err_keys = ["fro", "nuc", 1, 2]
        rel_errs = {}
        for key in rel_err_keys:
            rel_errs[key] = torch.linalg.norm(y - y_hat, dim=dim, ord=key) / torch.linalg.norm(y, dim=dim, ord=key)

        if idx % int(args.iters * args.prop_log) == 0:
            sl = (slice(None, 2),) + (slice(None, 3),) * (y.ndim - 1) if y.ndim > 2 else (slice(None, 10), 0)

            print(f"Size: {x.shape[-1]}x{x.shape[-1]}")

            print(f"Targets:\n{y[sl]} \nPredictions:\n{y_hat[sl]}\n")
            print(f"{args.loss} Loss: {loss:.3e} | Iter: {idx:,d} | LR: {sch.get_last_lr()[0]:.3e}")
            print(f"Out var: {out_var:.3e}\n")

            for key in rel_err_keys:
                print(f"Rel error {key}: {rel_errs[key].mean():.3e}")
            print("\n")

            if args.wandb:
                wandb.log({"iter": idx, "loss": loss, "lr": sch.get_last_lr()[0]})
                wandb.log({"out_var": out_var})
                for key in rel_err_keys:
                    wandb.log({f"relative error {str(key)}": rel_errs[key].mean()})

    toc = time.time()
    print_time_taken(toc - tic, text="Training time: ")

    if args.sch_name == "cos_grad":
        print(f"Decreased: {sch.did_update} | Increased: {sch.didnt_update}")
    print("\n")

    model.eval()
    # losses, rel_error_means = eval_model(model, test_loaderS, args=args, R=R, device=device)
    rel_error_means = eval_model(model, test_loaderS, args=args, R=R, device=device)
    evaled_matrices_rel = eval_OOD_matrices(model, args, device)

    if args.wandb:
        # for key, value in losses.items():
        #     wandb.summary[f"test_loss_{key}"] = value
        for key, value in rel_error_means.items():
            wandb.summary[f"test_rel_error_{key}"] = value
        for key, value in evaled_matrices_rel.items():
            wandb.summary[f"eval_rel_{key}"] = value
        wandb.finish()

    if args.save:
        outdir = Path(args.log_dir) / append_timestamp(f"run_{args.model}")
        outdir.mkdir(parents=True, exist_ok=True)
        pickle.dump(args.__dict__, open(outdir / "info.pkl", mode="wb"))
        model.to("cpu")
        torch.save(model.state_dict(), outdir / "weights.pth")
        print(f"Model saved {outdir}")


if __name__ == "__main__":
    parser = get_training_parser()
    args = parser.parse_args()
    main(args)
