import numpy as np
import torch
import torch.nn as nn

from nn.common import RecurrentConvolutionConfig


class RecurrentConvolution(nn.Module):
    def __init__(self, config: RecurrentConvolutionConfig):
        super().__init__()
        self.kernel_size = config.kernel_size
        self.target_size = config.target_size
        self.in_channels = 1
        self.channel_count = config.channel_count
        self.conv_in = nn.Conv2d(self.in_channels, self.channel_count, self.kernel_size, stride=2)
        self.conv_rec = nn.Conv2d(self.channel_count, self.channel_count, self.kernel_size, stride=2)
        self.conv_out = nn.Conv2d(self.channel_count, self.in_channels, self.kernel_size, stride=2)

        if config.channel_count == 1:
            self.conv_in = self.conv_rec  # point to same convolutional operator
            self.conv_out = self.conv_rec

        print(f"Params: {self.get_num_params()}")

    def forward(self, x):
        inp_size = x.shape[1]
        iters = int(np.log2(inp_size / self.target_size))
        x = x.unsqueeze(1)

        x = self.conv_in(x)

        for _ in range(iters - 2):
            x = self.conv_rec(x)

        x = self.conv_out(x)

        x = x.squeeze(1)
        x = x @ x.transpose(1, 2)
        x = torch.linalg.eigvalsh(x)[:, -1].unsqueeze(1)
        return x

    def get_num_params(self):
        num_params = sum(p.numel() for p in self.parameters())
        print(f"Number of parameters: {num_params}")
        return num_params
