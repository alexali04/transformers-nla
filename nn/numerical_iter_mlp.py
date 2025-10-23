from dataclasses import dataclass

import torch
import torch.nn as nn

from nn.common import LayerNorm


class MLPSeq(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_seq, config.n_seq, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(config.n_seq, config.n_seq, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # x: [B,N,D]
        x = self.c_fc(x.transpose(-1, -2))
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x.transpose(-1, -2)


class MLPDim(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # x: [B,N,D]
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class BlockMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp_seq = MLPSeq(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp_dim = MLPDim(config)

    def forward(self, x):
        x = x + self.mlp_seq(self.ln_1(x))
        x = x + self.mlp_dim(self.ln_2(x))
        return x


class IterNumericalMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_layer = config.n_layer

        self.input_proj = nn.Linear(1, config.n_embd, bias=False)
        self.mlp = nn.ModuleDict(
            dict(
                drop=nn.Dropout(config.dropout),
                block=BlockMLP(config),
                head=LayerNorm(config.n_embd, bias=config.bias),
            )
        )
        self.seq_mixer = nn.Linear(config.n_embd, 1, bias=False)
        self.dim_mixer = nn.Linear(config.n_seq, 1, bias=False)

        self.apply(self._init_weights)
        print(f"Parameters: {self.get_num_params() / 1e6:.2f} M")

    def get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, x):
        # x: [B, N, N]
        x = self.input_proj(x.reshape(x.shape[0], -1)[:, :, None])  # [B, T, D]
        x = self.mlp.drop(x)
        for _ in range(self.n_layer):
            x = self.mlp.block(x)  # [B, T, D]
        x = self.mlp.head(x)  # [B, T, D]
        x = self.seq_mixer(x).squeeze()  # [B, T]
        x = self.dim_mixer(x).squeeze()  # [B, 1]
        return x


@dataclass
class IterNumericalMLPConfig:
    n_layer: int = 6
    n_embd: int = 512
    n_seq: int = 5 * 5
    dropout: float = 0.0
    bias: bool = True
