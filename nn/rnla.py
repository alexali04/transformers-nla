import torch
import torch.nn as nn
from torch.linalg import qr

from nn.common import AttnConfig, Block, get_num_params


class RNLA(nn.Module):
    def __init__(self, config: AttnConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Linear(config.n_seq, config.n_embd)

        self.transformer = dict(
            drop=nn.Dropout(config.dropout),
            blocks=nn.ModuleList([Block(config) for _ in range(config.k_layer_loop)]),
        )

        self.transformer = nn.ModuleDict(self.transformer)
        self.num_params = get_num_params(self)
        print(f"Parameters: {self.num_params / 1e6:.2f} M")

    def forward(self, A, R: int):
        # A: [B,N,N]
        Omega = self.embed(A)  # [B,N,D]
        Omega = self.transformer.drop(Omega)
        for _ in range(R):
            for block in self.transformer.blocks:
                Omega = block(Omega)  # [B, N, D]
        Q, _ = qr(Omega, mode="reduced")
        C = Q.transpose(-1, -2) @ A
        _, Sigma, _ = torch.linalg.svd(C)
        return Sigma


class Noise(RNLA):
    def __init__(self, config: AttnConfig):
        super().__init__(config=config)
        self.embed = nn.Linear(config.n_embd, config.n_embd)

    def forward(self, A, R: int):
        # A: [B,N,N]
        Omega = torch.randn(A.shape[:2] + (self.config.n_embd,), device=A.device, dtype=A.dtype)  # [B,N,D]
        Omega = self.embed(Omega)  # [B,N,D]
        Omega = self.transformer.drop(Omega)
        for _ in range(R):
            Omega = self.transformer.block(Omega)  # [B,N,D]
        Q, _ = qr(Omega, mode="reduced")
        A_hat = Q @ Q.transpose(-1, -2) @ A
        return A_hat, Q, Omega


class RFSVD(Noise):
    def __init__(self, config: AttnConfig):
        super().__init__(config=config)
        self.embed = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, A, R: int):
        # A: [B,N,N]
        Omega = torch.randn(A.shape[:2] + (self.config.n_embd,), device=A.device, dtype=A.dtype)  # [B,N,D]
        A1 = A @ Omega
        A2 = A @ A1
        A3 = A @ A1
        Omega = torch.concat((Omega, A1, A2, A3), dim=-1)
        Omega = self.embed(Omega)  # [B,N,D]
        Omega = self.transformer.drop(Omega)
        for _ in range(R):
            for block in self.transformer.blocks:
                Omega = block(Omega)  # [B, N, D]

        Omega = A @ Omega
        Q, _ = qr(Omega, mode="reduced")
        C = Q.transpose(-1, -2) @ A
        _, Sigma, _ = torch.linalg.svd(C)
        return Sigma
