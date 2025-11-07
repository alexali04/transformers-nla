import pickle
import time
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import cola
import torch

from data.krylov_data import get_data
from krylov.krylov_fns import run_krylov
from nn.nn_fns import get_model
from utils.parsers import get_krylov_parser
from utils.timing import append_timestamp, print_time_taken


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {"single": torch.float32, "double": torch.float64}[args.precision]
    model_args = pickle.load(open(f"{args.model_path}/info.pkl", mode="rb"))
    _args = SimpleNamespace(**{**model_args, **args.__dict__})
    model_args = SimpleNamespace(**model_args)
    setattr(model_args, "out_dim", None)

    A, rhs, soln = get_data(_args, device=device, dtype=dtype)
    net = get_model(model_args)
    net.load_state_dict(torch.load(f"{args.model_path}/weights.pth"))
    net.eval()
    net.to(device, dtype)
    net = partial(net, R=model_args.n_layer)

    results = {args.alg: {"cold": {}, "warm": {}}}
    AOp, P = cola.ops.Dense(A), cola.ops.I_like(A)
    alg_kwargs = dict(A=AOp, b=rhs, tol=0, preconditioner=P, pbar=False)
    tic = time.time()

    x0 = torch.zeros_like(rhs)
    res, times = run_krylov(args.alg, alg_kwargs, x0=x0, soln=rhs, max_iters=args.iters)
    results[args.alg]["cold"]["res"] = res
    results[args.alg]["cold"]["times"] = times

    if model_args.op == "solve":
        xw = torch.cat([A[None], rhs[None]], dim=-1)
        xw = net(xw)[0].detach()[..., None]
    else:
        xw = (net(A[None])[0] @ rhs).detach()
    # alg_kwargs["preconditioner"] = cola.ops.Dense(net(A[None])[0].detach())
    res, times = run_krylov(args.alg, alg_kwargs, x0=xw, soln=rhs, max_iters=args.iters)
    results[args.alg]["warm"]["res"] = res
    results[args.alg]["warm"]["times"] = times

    toc = time.time()
    print_time_taken(toc - tic, text="Training time: ")

    if args.save:
        outdir = Path(args.log_dir) / append_timestamp(f"{args.alg}_{args.case}")
        outdir.mkdir(parents=True, exist_ok=True)
        pickle.dump(_args.__dict__, open(outdir / "info.pkl", mode="wb"))
        pickle.dump(results, open(outdir / "results.pkl", mode="wb"))
        print(f"Model saved {outdir}")


if __name__ == "__main__":
    parser = get_krylov_parser()
    args = parser.parse_args()
    main(args)
