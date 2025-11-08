import pickle
from copy import deepcopy
from types import SimpleNamespace

import torch

import nn


def get_model_from_path(search_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_args = pickle.load(open(f"{search_path}/info.pkl", mode="rb"))
    model_args = SimpleNamespace(**model_args)
    setattr(model_args, "out_dim", None)

    model = get_model(model_args)
    model.load_state_dict(torch.load(f"{search_path}/weights.pth"))
    model.to(device)
    model.eval()
    return model, model_args


def get_model(args):
    config = _construct_config(args)
    try:
        model = getattr(nn, args.model)(config)
    except AttributeError:
        raise ValueError(f"Model {args.model} not found. Supported models: {nn.supported_models}")
    return model


def _construct_config(args):
    if args.model == "RecurrentConvolution":
        config = getattr(nn, "RecurrentConvolutionConfig")(
            kernel_size=args.kernel_size,
            target_size=args.target_size,
            channel_count=args.channel_count,
        )
    else:
        config = getattr(nn, "AttnConfig")(
            n_head=args.n_head,
            n_embd=args.n_embd,
            n_seq=args.n_seq,
            dropout=args.dropout,
            pos_enc=args.pos_enc,
            attn_pattern=args.attn_pattern,
            n_layer=args.n_layer,
            max_seq=args.max_seq,
            bias=args.bias,
            input_inj=args.input_inj,
            k_layer_loop=args.k_layer_loop,
            mlp_act=args.mlp_act,
            mlp_n_layer=args.mlp_n_layer,
            mlp_factor=args.mlp_factor,
            out_dim=args.out_dim,
        )

    return config


RECURRENT_MODELS = [
    "RangeFormer",
    "HPFormer",
    "NumLoopFormer",
    "NumFormer",
    "RangeSolveFormer",
    "RangeInvFormer",
    "NumInvLoopFormer",
    "LinRangeN",
    "RandRangeFormer",
    "RNLA",
    "Noise",
    "RFSVD",
    "TTCFormer",
    "NumFormer",
]


def forward_pass(args, model, x, R=None):
    if args.model in RECURRENT_MODELS:
        out = model(x, R)
        y_hat = out[0] if isinstance(out, tuple) else out
        return y_hat

    return model(x)


def _delchainattr(obj, attr):
    attributes = attr.split(".")
    for a in attributes[:-1]:
        obj = getattr(obj, a)
    try:
        delattr(obj, attributes[-1])
    except AttributeError:
        raise


def unflatten_like(vector, likeTensorList):
    outList = []
    i = 0
    for tensor in likeTensorList:
        n = tensor.numel()
        outList.append(vector[i : i + n].view(tensor.shape))
        i += n
    return outList


def _setchainattr(obj, attr, value):
    attributes = attr.split(".")
    for a in attributes[:-1]:
        obj = getattr(obj, a)
    setattr(obj, attributes[-1], value)


class SubspaceModel(torch.nn.Module):
    def __init__(self, net, target_dim: int = 32, seed=None, device=None):
        super().__init__()

        self.d = target_dim
        self._forward_net = [net]
        initnet = deepcopy(net)

        for orig_name, orig_p in initnet.named_parameters():
            if orig_p.requires_grad:
                _delchainattr(net, orig_name)

        aux = [(n, p) for n, p in initnet.named_parameters() if p.requires_grad]
        self.names, self.trainable_initparams = zip(*aux)
        self.trainable_initparams = [param.to(device) for param in self.trainable_initparams]
        self.names = list(self.names)

        self.D = sum([param.numel() for param in self.trainable_initparams])
        self.subspace_params = torch.nn.Parameter(torch.zeros(self.d))
        if seed is not None:
            torch.manual_seed(seed)

        self.P = torch.randn(self.D, self.d, device=device)
        self.P, _ = torch.linalg.qr(self.P, mode="reduced")

    def _to(self, *args, **kwargs):
        self._forward_net[0].to(*args, **kwargs)
        self.trainable_initparams = [param.to(*args, **kwargs) for param in self.trainable_initparams]
        return super().to(*args, **kwargs)

    def forward(self, *args, **kwargs):
        flat_projected_params = self.P @ self.subspace_params

        unflattened_params = unflatten_like(flat_projected_params, self.trainable_initparams)
        iterables = zip(self.names, self.trainable_initparams, unflattened_params)
        for p_name, init, proj_param in iterables:
            p = init + proj_param.view(*init.shape)
            _setchainattr(self._forward_net[0], p_name, p)

        return self._forward_net[0](*args, **kwargs)

    def get_state_dict(self):
        flat_projected_params = self.P @ self.subspace_params

        unflattened_params = unflatten_like(flat_projected_params, self.trainable_initparams)
        iterables = zip(self.names, self.trainable_initparams, unflattened_params)

        state_dict = self._forward_net[0].state_dict()

        for p_name, init, proj_param in iterables:
            p = init + proj_param.view(*init.shape)
            state_dict[p_name] = p

        return state_dict
