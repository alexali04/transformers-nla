import pickle
import time
from pathlib import Path

import numpy as np
import torch
import wandb
from torch.nn.utils import clip_grad_norm_

from data.data_fns import get_loaders
from eval.eval_fns import compute_rel_errs, eval_model, eval_OOD_matrices, print_stats
from loss.loss_fns import get_loss, pre_process
from nn.nn_fns import forward_pass, get_model
from opt.opt_fns import get_optimizer
from opt.scheduler_fns import get_scheduler
from utils.parsers import clean_training_parser, get_training_parser
from utils.sampler import get_recurrence_length
from utils.training import seed_everything
from utils.timing import append_timestamp, print_time_taken


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float64 if args.pres == "double" else torch.float32
    if args.wandb:
        wandb.init(project=args.wandb_proj)
        wandb.config.update(args.__dict__)

    if args.seed >= 0:
        seed_everything(seed=args.seed)

    train_loaderS, test_loaderS = get_loaders(args)
    ds_idxS = [idx for idx in range(len(train_loaderS))]
    model = get_model(args)
    opt = get_optimizer(args.optimizer, model.parameters(), lr=args.lr, wd=args.wd, args=args)
    loss_fn = get_loss(args.loss)

    tic = time.time()
    model.to(device, dtype)
    model.train()
    Rs = get_recurrence_length(args)
    sch = get_scheduler(opt=opt, args=args, model=model, train_loader=train_loaderS, device=device, R=Rs[0])
    print(f"Model Type: {args.model} | Iters: {args.iters} | Scheduler: {args.sch_name} | Optimizer: {args.optimizer}")

    for idx in range(args.iters):
        batch = next(train_loaderS[np.random.choice(ds_idxS)])
        x, y = batch[0].to(device, dtype), batch[1].to(device, dtype)
        R_i = Rs[idx] if Rs is not None else None
        y_out = forward_pass(args, model, x, R_i)
        y, y_hat = pre_process(y, y_out, x, args.loss)
        loss = loss_fn(y_hat, y)
        loss.backward()
        grad_norm = clip_grad_norm_(model.parameters(), max_norm=args.grad_norm)
        opt.step()
        opt.zero_grad()
        sch.step()

        rel_errs = compute_rel_errs(y=y, y_hat=y_hat)
        if idx % int(args.iters * args.prop_log) == 0:
            out = print_stats(args, x, y_hat, y, loss, idx, sch.get_last_lr()[0], rel_errs, grad_norm)
            regular_stats, additional_stats = out
            if args.wandb:
                wandb.log(regular_stats)
                wandb.log(additional_stats)

    toc = time.time()
    print_time_taken(toc - tic, text="Training time: ")
    model.eval()
    test_rel_errS, losses = eval_model(model, test_loaderS, args=args, R=args.n_layer, device=device, dtype=dtype)
    ood_rel_errS, ood_losses = eval_OOD_matrices(model, args, R=args.n_layer, device=device, dtype=dtype)

    if args.wandb:
        if args.op != "inv":
            for key, value in losses.items():
                wandb.summary[f"test_loss_{key}"] = value
        for key, value in test_rel_errS.items():
            wandb.summary[f"test_rel_err_{key}"] = value
        for key, value in ood_rel_errS.items():
            wandb.summary[f"eval_{key}"] = value
        for key, value in ood_losses.items():
            wandb.summary[f"eval_{key}"] = value
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
    clean_training_parser(args)
    main(args)
