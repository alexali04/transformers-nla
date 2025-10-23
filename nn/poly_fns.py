import torch
import torch.nn as nn


class Short(nn.Module):
    def __init__(
        self,
        n_embd: int,
        max_n_seq: int,
        use_res: bool,
        use_causal: bool,
    ):
        super().__init__()
        self.use_res = use_res

        self.conv = nn.Conv1d(
            in_channels=n_embd,
            out_channels=n_embd,
            kernel_size=max_n_seq,
            groups=n_embd,
            padding=max_n_seq - 1 if use_causal else max_n_seq // 2,
        )

    def forward(self, x: torch.Tensor):
        n_seq = x.size(1)
        y = self.conv(x.transpose(1, 2))[..., :n_seq].transpose(1, 2)
        if self.use_res:
            y = y + x
        return y


class BaseConv(nn.Module):
    def __init__(
        self,
        n_embd: int,
        n_seq: int,
        conv_type: str,
        use_res: bool,
        use_causal: bool,
        bias: bool,
    ):
        super().__init__()

        self.n_embd = n_embd
        self.n_seq = n_seq
        self.conv_type = conv_type
        self.use_res = use_res
        self.use_causal = use_causal

        self.projection = nn.Linear(self.n_embd, self.n_embd, bias=bias)
        self.in_proj = nn.Linear(self.n_embd, self.n_embd, bias=bias)
        self.out_proj = nn.Linear(self.n_embd, self.n_embd, bias=bias)

        if self.use_res:
            self.branch_weights = nn.Parameter(torch.zeros(2), requires_grad=True)
            self.branch_weights.data[0] = 1

        self.conv = Short(n_embd, max_n_seq=n_seq, use_res=use_res, use_causal=use_causal)

    def forward(self, x):
        # x: [B,N,D]
        u_conv = self.conv(self.in_proj(x))
        u_proj = self.projection(x)
        y = self.out_proj(u_conv * u_proj)
        if self.use_res:
            y = self.branch_weights[0] * y + self.branch_weights[1] * x
        return y  # [B,N,D]


class ConvMixer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.base_conv = BaseConv(
            n_embd=config.n_embd,
            n_seq=config.n_seq,
            conv_type=config.attn_pattern,
            use_res=True,
            use_causal=True,
            bias=config.bias,
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.base_conv(x)
        x = self.dropout(x)
        return x
