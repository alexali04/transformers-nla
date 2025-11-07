import torch
import torch.nn as nn

from nn.common import POSENC, AttnConfig, Block, LayerNorm, get_num_params


class RangeSolveFormer(nn.Module):
    def __init__(self, config: AttnConfig):
        super().__init__()
        self.max_seq = config.max_seq
        self.pos_enc = POSENC[config.pos_enc](n_seq=config.max_seq, n_embd=config.n_embd)
        init_embd = self.pos_enc.change_embedding(config.n_embd)
        self.encoder = nn.Linear(config.max_seq, init_embd, bias=config.bias)
        config.n_embd = config.n_embd + 1  # redefinition that affects below only
        self.transformer = nn.ModuleDict(
            dict(
                drop=nn.Dropout(config.dropout),
                block=Block(config),
                ln=LayerNorm(config.n_embd, bias=config.bias),
            )
        )
        self.dim_mixer = nn.Linear(config.n_embd, 1, bias=config.bias)
        print(f"Parameters: {get_num_params(self) / 1e6:.2f} M")

    def pad_x(self, x):
        pad_len = self.max_seq - x.shape[-1]
        pad_mask = torch.zeros(*x.shape[:-1], pad_len, device=x.device)
        x = torch.cat([x, pad_mask], dim=-1)
        return x

    def forward(self, x, R):
        rhs = x[..., [-1]]
        x = x[..., :-1]

        x = self.pad_x(x)  # [B, N, N_max]
        x = self.encoder(x)  # [B, N, D]
        x = self.pos_enc(x)  # [B, N, D]
        x = torch.cat([x, rhs], dim=-1)  # [B, N, D+1]

        x = self.transformer.drop(x)
        for _ in range(R):
            x = self.transformer.block(x)  # [B, N, D+1]

        x = self.transformer.ln(x)
        x = self.dim_mixer(x)  # [B, N, 1]
        return x[..., 0]
