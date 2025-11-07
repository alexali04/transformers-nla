import pickle
import time
from functools import partial
from types import SimpleNamespace

import torch
import wandb

from data.data_fns import get_loaders
from nn.gp_fn import gp_net_logdet_quad, gp_predict, ker_fn
from nn.nn_fns import get_model
from opt.opt_fns import get_optimizer
from utils.parsers import get_gp_parser
from utils.timing import append_timestamp, print_time_taken


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    search_path = f"./saved_runs/{args.model_path}"
    model_args = pickle.load(open(f"{search_path}/info.pkl", mode="rb"))
    model_args = SimpleNamespace(**model_args)

    if args.wandb:
        wandb.init(project=args.wandb_proj)
        wandb.config.update(args.__dict__)

    dataset = f"{args.operation}_{args.distribution}_{args.description}"
    loaders = get_loaders(dataset, args.datapath, args.bsz)
    train_loader, test_loader = loaders["train"], loaders["test"]
    net = get_model(model_args)
    net.load_state_dict(torch.load(f"{search_path}/weights.pth"))
    net.eval()
    net.to(device)
    net = partial(net, R=model_args.n_layer)
    ls = torch.tensor([0.1], device=device).requires_grad_(True)
    # log_sigma = torch.tensor([-1.5], device=device).requires_grad_(True)
    log_sigma = torch.tensor([-2.0], device=device)
    params = [ls, log_sigma]
    opt = get_optimizer(args.optimizer, [ls], lr=args.lr, wd=args.wd)

    tic = time.time()
    for idx, batch in zip(range(args.iters), train_loader):
        x, y = batch[0].to(device), batch[1].to(device)
        x, y = x[0], y[0]
        K = ker_fn(x, params)
        loss, *_ = gp_net_logdet_quad(K, y[..., None], net, args.vtol, args.case)
        loss.backward()
        opt.step()
        opt.zero_grad()

        if idx % int(args.iters * args.prop_log) == 0:
            real_loss = gp_net_logdet_quad(K, y[..., None], net, args.vtol, "actual")
            text = f"Loss: {real_loss:.2e} | Iter: {idx:,d}"
            text += f" | ls: {ls.item():.2e} | sigma: {torch.exp(log_sigma).item():.2e}"
            print(text)
            if args.wandb:
                wandb.log({"iter": idx, "loss": real_loss, "ls": ls.item()})
                wandb.log({"log_sigma": log_sigma.item(), "sigma": torch.exp(log_sigma).item()})

    toc = time.time()
    print_time_taken(toc - tic, text="Training time: ")

    for idx, batch in zip(range(1), test_loader):
        x_test, y_test = batch[0].to(device), batch[1].to(device)
        x_test, y_test = x_test[0], y_test[0]
        mu, _ = gp_predict(x_test, x_train=x, y_train=y[:, None], params=params)
        test_rmse = torch.sqrt(torch.mean((y_test - mu) ** 2.))
        print(f"Test RMSE: {test_rmse:.3e} | Iter: {idx:,d}")

    if args.wandb:
        wandb.summary["test_rmse"] = test_rmse
        wandb.finish()


if __name__ == "__main__":
    parser = get_gp_parser()
    args = parser.parse_args()
    main(args)
