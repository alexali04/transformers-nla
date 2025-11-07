import torch
import torch.nn as nn

from nn.common import POSENC, AttnConfig, Block, LayerNorm, get_num_params


class RangeFormer(nn.Module):
    def __init__(self, config: AttnConfig):
        super().__init__()
        self.max_seq = config.max_seq
        self.input_inj = config.input_inj

        self.pos_enc = POSENC[config.pos_enc](n_seq=self.max_seq, n_embd=config.n_embd)
        init_embd = self.pos_enc.change_embedding(config.n_embd)
        self.encoder = nn.Linear(self.max_seq, init_embd, bias=config.bias)
        self.transformer = nn.ModuleDict(
            dict(
                drop=nn.Dropout(config.dropout),
                block=Block(config),
                ln=LayerNorm(config.n_embd, bias=config.bias),
            )
        )
        self.dim_mixer = nn.Linear(config.n_embd, 1, bias=config.bias)
        if config.out_dim == 1:
            self.seq_mixer = nn.Linear(self.max_seq, 1, bias=config.bias)
        else:
            self.seq_mixer = nn.Identity()
        print(f"Parameters: {get_num_params(self) / 1e6:.2f} M")

    def pad_x(self, x):
        N = x.shape[-1]
        pad_len = self.max_seq - N
        pad_mask = torch.zeros(*x.shape[:-1], pad_len, device=x.device)
        x = torch.cat([x, pad_mask], dim=-1)
        return x

    def forward(self, x, R):
        # x: [B, N, N]
        x = self.pad_x(x)  # [B, N, N_max]
        x = self.encoder(x)  # [B, N, D]
        x = self.pos_enc(x)
        x0 = x.clone()

        x = self.transformer.drop(x)
        for _ in range(R):
            x = self.transformer.block(x)
            if self.input_inj:
                x = x + x0

        x = self.transformer.ln(x)  # [B, N, D]
        x = self.dim_mixer(x)[..., 0]  # [B, N]
        x_seq = self.pad_x(x)  # [B, N_max]
        x = self.seq_mixer(x_seq)  # [B, O]

        return x
