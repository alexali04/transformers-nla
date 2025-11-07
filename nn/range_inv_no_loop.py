import torch
import torch.nn as nn

from nn.common import POSENC, AttnConfig, Block, LayerNorm, get_num_params


class RangeInvNoLoopFormer(nn.Module):
    """
    Range InvFormer
    """

    def __init__(self, config: AttnConfig):
        super().__init__()
        self.config = config
        config.n_seq = config.max_seq
        self.n_seq = config.n_seq

        self.pos_enc = POSENC[config.pos_enc](n_seq=self.n_seq, n_embd=config.n_embd)
        init_embd = self.pos_enc.change_embedding(config.n_embd)
        self.encoder = nn.Linear(config.n_seq, init_embd, bias=config.bias)

        self.transformer = nn.ModuleDict(
            dict(
                drop=nn.Dropout(config.dropout),
                blocks=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln=LayerNorm(config.n_embd, bias=config.bias),
            )
        )

        self.dim_mixer = nn.Linear(config.n_embd, config.n_seq, bias=config.bias)

        print(f"Parameters: {get_num_params(self) / 1e6:.2f} M")

        # self.apply(self._init_weights)
        # for pn, p in self.named_parameters():
        #     if pn.endswith("output_proj.weight"):
        #         torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def pad_x(self, x):
        N = x.shape[-1]
        pad_len = self.config.max_seq - N
        pad_mask = torch.zeros(*x.shape[:-1], pad_len, device=x.device)
        x = torch.cat([x, pad_mask], dim=-1)
        return x

    def forward(self, x):
        N = x.shape[-1]
        x = self.pad_x(x)  # [B, N, N_max]
        x = self.encoder(x)  # [B, N, D]
        x = self.pos_enc(x)

        x = self.transformer.drop(x)
        for block in self.transformer.blocks:
            x = block(x)  # [B, N, D]

        x = self.transformer.ln(x)
        x = self.dim_mixer(x)  # [B, N, N_max]
        x = x[..., :N]  # [B, N, N]

        return x
