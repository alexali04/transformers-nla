import torch.nn as nn

from nn.common import POSENC, AttnConfig, Block, LayerNorm, get_num_params


class LinRangeN(nn.Module):
    def __init__(self, config: AttnConfig):
        """
        Range transformer for matrix to matrix problems
        """
        super().__init__()
        self.config = config

        self.pos_enc = POSENC[config.pos_enc](n_seq=config.n_seq, n_embd=config.n_embd)
        init_embd = self.pos_enc.change_embedding(config.n_embd)
        self.embed = nn.Linear(config.n_seq, init_embd)

        self.transformer = dict(
            drop=nn.Dropout(config.dropout),
            block=Block(config),
            ln=LayerNorm(config.n_embd, bias=config.bias),
        )

        self.transformer = nn.ModuleDict(self.transformer)

        self.head = nn.Linear(config.n_embd, config.n_seq, bias=False)
        self.num_params = get_num_params(self)
        print(f"Parameters: {self.num_params / 1e6:.2f} M")

    def forward(self, x, R):
        # x: [B,N,N]
        x = self.embed(x)  # [B,N,D]
        x = self.transformer.drop(x)
        for _ in range(R):
            x = self.transformer.block(x)  # [B,N,D]
        x = self.transformer.ln(x)
        x = self.head(x)  # [B,N,N]
        return x
