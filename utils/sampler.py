import numpy as np
import torch


def get_recurrence_length(args):
    if args.sample_rec == "poisson":
        Rs, _ = sample_recurrence_length(args.n_layer, sigma=1.0, num_samples=args.iters, max_n=30)
    elif args.sample_rec == "cons":
        Rs = torch.ones(args.iters).int() * args.n_layer
    else:
        raise ValueError(f"{args.sample_rec=} not found")
    return Rs


def sample_recurrence_length(n_layer, sigma, num_samples, max_n):
    target_mu = np.log(n_layer) - 0.5 * np.square(sigma)
    taus = np.random.normal(target_mu, sigma, num_samples)
    rs = np.random.poisson(np.exp(taus)) + 1
    # min_n = min(4, n_layer)
    min_n = 1
    rs = np.clip(rs, min_n, max_n)
    return rs, target_mu
