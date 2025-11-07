import pickle
from types import SimpleNamespace

import torch

from data.data_fns import get_loaders
from nn.common import get_num_params
from nn.nn_fns import get_model
from rnla.rnla_fns import construct_grid, plot_contour
from utils.parsers import get_eval_parser


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    search_path = f"./saved_runs/{args.model_path}"
    model_args = pickle.load(open(f"{search_path}/info.pkl", mode="rb"))
    model_args = SimpleNamespace(**model_args)

    dataset = f"{model_args.operation}_{model_args.distribution}_{model_args.description}"
    loaders, data_kwargs = get_loaders(dataset, model_args.datapath, args.eval_bsz)

    model = get_model(model_args)
    model.load_state_dict(torch.load(f"{search_path}/weights.pth"))
    model.to(device)
    model.eval()

    text = f"Model Class: {model_args.model} | Matrix dimension: {model_args.n_seq} | "
    text += f"Parameters: {get_num_params(model) / 1e6:.2f} M"
    print(text)

    for idx, batch in zip(range(1), loaders["train"]):
        A = batch[0].to(device)
        *_, Omega = model(A, R=model_args.n_layer)
    X = Omega.detach().cpu().numpy()
    grid = construct_grid(X, n_components=2, grid_args=(-5, 5, 100))
    plot_contour(grid, save_path="logs/")


if __name__ == "__main__":
    parser = get_eval_parser()
    args = parser.parse_args()
    main(args)
