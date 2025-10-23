import math

import torch
import torch.nn as nn

from nn.common import POSENC, AttnConfig, Block, LayerNorm, get_num_params


class TTCFormer(nn.Module):
    """
    Looped, flat transformer for test-time compute
    """

    def __init__(self, config: AttnConfig):
        super().__init__()
        self.config = config

        n_seq = config.n_seq * config.n_seq

        self.pos_enc = POSENC[config.pos_enc](n_seq=n_seq, n_embd=config.n_embd)
        init_embd = self.pos_enc.change_embedding(config.n_embd)
        self.embed = nn.Linear(1, init_embd, bias=False)

        self.transformer = dict(
            drop=nn.Dropout(config.dropout),
            block=Block(config),
            ln=LayerNorm(config.n_embd, bias=config.bias),
        )

        self.transformer = nn.ModuleDict(self.transformer)
        self.adapter = nn.Linear(2 * config.n_embd, config.n_embd, bias=False)

        self.dim_mixer = nn.Linear(config.n_embd, 1, bias=False)
        self.seq_mixer = nn.Linear(config.n_seq * config.n_seq, 1, bias=False)
        print(f"Parameters: {get_num_params(self) / 1e6:.2f} M")

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("output_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2))

        self.n_layer = config.n_layer

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, x, R):
        """
        h - hidden dimension, n_embd
        t - sequence dimension, n_seq
        """
        B, N, _ = x.size()

        # noise for "path independence" - [B, N^2, n_embd]
        s_i = torch.randn(B, N * N, self.config.n_embd).to(x.device) * (2 / 5) ** 0.5

        x = x.reshape(x.shape[0], -1)[:, :, None]  # [B, N^2, 1]
        x = self.embed(x)  # [B, N^2, n_embd]
        x = self.pos_enc(x)  # [B, N^2, n_embd]
        e = self.transformer.drop(x)

        # Recurrent block
        for _ in range(R):
            h = self.adapter(torch.cat([s_i, e], dim=-1))  # input injection
            s_i = self.transformer.block(h)

        # Coda
        x = self.transformer.ln(s_i)
        x = self.dim_mixer(x)[..., 0]
        x = self.seq_mixer(x)
        return x
