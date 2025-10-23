import math
from dataclasses import dataclass, field
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from nn.poly_fns import ConvMixer


def get_num_params(module):
    n_params = sum(p.numel() for p in module.parameters())
    return n_params

@dataclass
class AttnConfig:
    n_head: int
    n_embd: int
    n_seq: int
    dropout: float
    bias: bool
    pos_enc: str
    attn_pattern: str
    n_layer: int
    k_layer_loop: int
    mlp_act: str
    mlp_n_layer: int
    mlp_factor: int
    out_dim: int
    max_seq: int = None
    input_inj: bool = None
    norms: List[str] = field(default_factory=lambda: ["last"])

    def __post_init__(self):
        if self.max_seq is None:
            self.max_seq = self.n_seq
        if self.input_inj is None:
            self.input_inj = False
        if self.out_dim is None:
            self.out_dim = self.n_seq


@dataclass
class RecurrentConvolutionConfig:
    channel_count: int = 1
    kernel_size: int = 2
    target_size: int = 4


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.qkv_proj = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.output_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.attn = ATTN[config.attn_pattern]

        if config.pos_enc == "rope":
            assert config.n_embd // config.n_head % 2 == 0, "RoPE requires d_head divisible by 2"
            self.rope = RoPE(config.n_seq, config.n_embd // config.n_head)
        else:
            self.rope = None

    def forward(self, x):
        B, N, D = x.size()

        q, k, v = self.qkv_proj(x).split(self.n_embd, dim=2)
        k = k.view(B, N, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, N, hs)
        q = q.view(B, N, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, N, hs)
        v = v.view(B, N, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, N, hs)

        if self.rope is not None:
            q = self.rope(q)
            k = self.rope(k)

        y = self.attn(q=q, k=k, v=v)
        y = y.transpose(1, 2).contiguous().view(B, N, D)  # re-assemble all head outputs side by side
        y = self.resid_dropout(self.output_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        din, dout = config.n_embd, config.mlp_factor * config.n_embd
        self.c_fc = nn.Linear(din, dout, bias=config.bias)
        self.act = ACTs[config.mlp_act]
        self.inter = [nn.Linear(dout, dout, bias=config.bias) for _ in range(config.mlp_n_layer - 2)]
        self.inter = nn.ModuleList(self.inter)
        self.c_proj = nn.Linear(dout, din, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.act(self.c_fc(x))
        for layer in self.inter:
            x = self.act(layer(x))
        x = self.dropout(self.c_proj(x))
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_seq = LayerNorm(config.n_embd, bias=config.bias)
        if config.attn_pattern in ATTN.keys():
            self.seq_mix = Attention(config)
        elif config.attn_pattern in ["short", "conv"]:
            self.seq_mix = ConvMixer(config)
        else:
            raise ValueError(f"{config.attn_pattern=} not found")
        self.ln_dim = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.seq_mix(self.ln_seq(x))
        x = x + self.mlp(self.ln_dim(x))
        return x


class NoneEnc(nn.Module):
    def __init__(self, **_):
        super().__init__()

    @staticmethod
    def change_embedding(n_embd):
        return n_embd

    def forward(self, x):
        return x


class PosEncConcat(nn.Module):
    def __init__(self, **_):
        super().__init__()
        self.scale = nn.Parameter(torch.Tensor([1.0]))

    @staticmethod
    def change_embedding(n_embd):
        return n_embd - 1

    def forward(self, x):
        device = x.device
        B, S, _ = x.size()
        pos_enc = torch.arange(S, device=device)
        pos_encs = pos_enc.repeat(B, 1).unsqueeze(-1)
        pos_encs = self.scale * pos_encs
        return torch.cat([x, pos_encs], dim=-1)


class SineEnc(nn.Module):
    def __init__(self, n_seq: int, n_embd: int):
        super().__init__()
        pe = torch.zeros(n_seq, n_embd)  # [N,D]
        position = torch.arange(0, n_seq, dtype=torch.float).unsqueeze(1)  # [N,1]
        cons = -np.log(1e4) / n_embd
        div_term = torch.exp(torch.arange(0, n_embd, 2).float() * cons)  # [D//2,]
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1,N,D]
        self.register_buffer("pe", pe)

    @staticmethod
    def change_embedding(n_embd):
        return n_embd

    def forward(self, x):
        # x: [B,N,D]
        return x + self.pe[:, : x.size(1), :]


class LearnedEnc(nn.Module):
    def __init__(self, n_seq: int, n_embd: int):
        super().__init__()
        self.max_seq = n_seq
        self.position_embedding = nn.Embedding(self.max_seq, n_embd)

    @staticmethod
    def change_embedding(n_embd):
        return n_embd

    def forward(self, x):
        # x: [B,N,D]
        B, S, _ = x.size()
        pos = torch.arange(S, dtype=torch.long, device=x.device)  # [S]
        pos_emb = self.position_embedding(pos).unsqueeze(0)  # [1, S, D]
        pos_emb = pos_emb.expand(B, -1, -1)

        return x + pos_emb


class RoPE(nn.Module):
    def __init__(self, n_seq: int, n_embd: int):
        super().__init__()
        self.n_embd = n_embd
        self.n_seq = n_seq

        half_dim = self.n_embd // 2

        emb = torch.exp(
            torch.arange(0, half_dim, dtype=torch.float) * -math.log(1e4) / (half_dim - 1)  # [D//2]
        )
        pos = torch.arange(self.n_seq, dtype=torch.float).unsqueeze(1)  # [N, 1]

        self.register_buffer("cos_cached", torch.cos(pos * emb))  # [N, D//2]
        self.register_buffer("sin_cached", torch.sin(pos * emb))  # [N, D//2]

    @staticmethod
    def change_embedding(n_embd):
        return n_embd

    def forward(self, x):
        S = x.size(1)

        cos = self.cos_cached[:S]  # [N, D//2]
        sin = self.sin_cached[:S]  # [N, D//2]

        if len(x.shape) == 4:  # [B, num_heads, S, D//num_heads]
            B, NH, S, D_per_head = x.shape

            x = x.reshape(B, NH, S, D_per_head // 2, 2)  # [B, NH, S, D//(2 * NH), 2]
            cos = cos.unsqueeze(0).unsqueeze(2)
            sin = sin.unsqueeze(0).unsqueeze(2)

            x_even = x[..., 0]  # [B, num_heads, S, D//(num_heads * 2)]
            x_odd = x[..., 1]  # [B, num_heads, S, D//(num_heads * 2)]

            rot_x_even = x_even * cos - x_odd * sin
            rot_x_odd = x_even * sin + x_odd * cos

            x = torch.stack([rot_x_even, rot_x_odd], dim=-1)
            rotated_x = torch.stack([rot_x_even, rot_x_odd], dim=-1)

            rotated_x = rotated_x.reshape(B, NH, S, D_per_head)

        else:
            B, S, D = x.shape
            x = x.reshape(B, S, D // 2, 2)

            cos = cos.unsqueeze(0)  # [1, N, D//2]
            sin = sin.unsqueeze(0)  # [1, N, D//2]

            x_even = x[..., 0]  # [B, N, D//2]
            x_odd = x[..., 1]  # [B, N, D//2]

            rot_x_even = x_even * cos - x_odd * sin
            rot_x_odd = x_even * sin + x_odd * cos

            rotated_x = torch.stack([rot_x_even, rot_x_odd], dim=-1)
            rotated_x = rotated_x.reshape(B, S, D)

        return rotated_x


ACTs = {
    "relu": nn.ReLU(),
    "silu": nn.SiLU(),
    "gelu": nn.GELU(),
}


POSENC = {
    "none": NoneEnc,
    "sine": SineEnc,
    "concat": PosEncConcat,
    "learned": LearnedEnc,
    "rope": RoPE,
}


def linear_elu(q, k, v):
    Q = 1.0 + F.elu(q)
    K = 1.0 + F.elu(k)
    y = Q @ (K.transpose(-2, -1) @ v)
    return y


def linear_taylor(q, k, v):
    Q_0 = torch.ones_like(q)
    Q_1 = q
    Q_2 = q * q / 2.0

    K_0 = torch.ones_like(k)
    K_1 = k
    K_2 = k * k / 2.0

    Q = Q_0 + Q_1 + Q_2
    K = K_0 + K_1 + K_2
    y = Q @ (K.transpose(-2, -1) @ v)
    return y


def quadratic(q, k, v):
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    att = F.softmax(att, dim=-1)
    y = att @ v
    return y


ATTN = {
    "linear_taylor": linear_taylor,
    "linear_elu": linear_elu,
    "quadratic": quadratic,
}

# none --> ConvMixer
