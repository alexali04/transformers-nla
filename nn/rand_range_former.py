import torch
import torch.nn as nn

from nn.common import POSENC, AttnConfig, Block, LayerNorm, get_num_params


class RandRangeFormer(nn.Module):
    def __init__(self, config: AttnConfig):
        super().__init__()
        self.config = config
        self.do_qr = False

        self.pos_enc = POSENC[config.pos_enc](n_seq=config.n_seq, n_embd=config.n_embd)
        self.init_embd = self.pos_enc.change_embedding(config.n_embd)
        self.encoder = nn.Linear(config.n_embd, config.n_embd)
        self.transformer = dict(
            drop=nn.Dropout(config.dropout),
            blocks=nn.ModuleList([Block(config) for _ in range(config.k_layer_loop)]),
        )
        self.transformer = nn.ModuleDict(self.transformer)
        self.last_ln = LayerNorm(config.n_embd, bias=config.bias) if "last" in config.norms else nn.Identity()
        self.dim_mixer = nn.Linear(config.n_embd, 1, bias=False)

        print(f"Parameters: {get_num_params(self) / 1e6:.2f} M")

    def forward(self, x, R):
        Omega = torch.randn(x.shape[-1], self.init_embd, device=x.device)
        if self.do_qr:
            Omega, _ = torch.linalg.qr(Omega, mode="reduced")
        x_embd = x @ Omega
        x_embd = self.pos_enc(x_embd)
        x_embd = self.encoder(x_embd)

        x = self.transformer.drop(x_embd)
        for _ in range(R):
            for block in self.transformer.blocks:
                x = block(x)  # [B, N, D]

        x = self.last_ln(x)  # [B,N,D]
        x = self.dim_mixer(x)[..., 0]  # [B,N]

        return x
