import torch
import torch.nn as nn

from nn.common import POSENC, AttnConfig, Block, LayerNorm, get_num_params


class NumFormer(nn.Module):
    def __init__(self, config: AttnConfig):
        super().__init__()
        self.n_seq = config.max_seq * config.max_seq
        self.pos_enc = POSENC[config.pos_enc](n_seq=self.n_seq, n_embd=config.n_embd)
        init_embd = self.pos_enc.change_embedding(config.n_embd)
        self.encoder = nn.Linear(1, init_embd, bias=config.bias)

        self.transformer = nn.ModuleDict(
            dict(
                drop=nn.Dropout(config.dropout),
                blocks=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln=LayerNorm(config.n_embd, bias=config.bias),
            )
        )

        self.dim_mixer = nn.Linear(config.n_embd, 1, bias=config.bias)
        self.out_dims = slice(-config.out_dim, None, 1)
        print(f"Parameters: {get_num_params(self) / 1e6:.2f} M")

    def pad_x(self, x):
        N = x.shape[1]
        pad_len = self.n_seq - N
        pad_mask = torch.zeros(size=(x.shape[0], pad_len, x.shape[-1]), device=x.device)
        x = torch.cat([x, pad_mask], dim=1)
        return x

    def forward(self, x, R):
        x = x.reshape(x.shape[0], -1)[:, :, None]  # [B,N^2,1]
        x = self.pad_x(x)

        x = self.encoder(x)  # [B,N^2,D]
        x = self.pos_enc(x)  # [B,N^2,D*]
        x = self.transformer.drop(x)

        R = R if R <= len(self.transformer.blocks) else len(self.transformer.blocks)
        for rdx in range(R):
            x = self.transformer.blocks[rdx](x)  # [B,N^2,D]

        x = self.transformer.ln(x)  # [B,N^2,D]
        x = self.dim_mixer(x)[..., 0]  # [B,N^2]
        x = x[..., self.out_dims]
        return x
    
    def get_encodings(self, x):
        x = x.reshape(x.shape[0], -1)[:, :, None]  # [B,N^2,1]
        x = self.pad_x(x)

        x = self.encoder(x)  # [B,N^2,D]
        return x
    
    def get_encoding_layer(self):
        return self.encoder

