import math
from typing import List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def fft_conv(
    u,
    k,
    dropout_mask,
    gelu=True,
    k_rev=None,
    causal=True,
    resid=True,
    do_kfft=True,
):
    seqlen = u.shape[-1]

    if causal:
        fft_size = 2 * seqlen
    else:
        fft_size = seqlen

    if do_kfft:
        u_f = torch.fft.rfft(u.to(dtype=k.dtype), n=fft_size)
        k_f = torch.fft.rfft(k[..., :seqlen], n=fft_size) / fft_size
        if k_rev is not None:
            k_rev_f = torch.fft.rfft(k_rev, n=fft_size) / fft_size
            k_f = k_f + k_rev_f.conj()
    else:
        u_f = torch.fft.rfft(u, n=fft_size)
        k_f = k[: u_f.shape[1], : u_f.shape[2]].to(device=u.device)

    if len(u.shape) > 3:
        k_f = k_f.unsqueeze(1)
    y = torch.fft.irfft(u_f * k_f, n=fft_size, norm="forward")[..., :seqlen]

    if resid:
        out = y + u
    else:
        out = y
    if gelu:
        out = F.gelu(out)
    if dropout_mask is not None:
        return (out * rearrange(dropout_mask, "b H -> b H 1")).to(dtype=u.dtype)
    else:
        return out.to(dtype=u.dtype)


class ShortConvolution(nn.Module):
    """
    Simple wrapper around nn.Conv1d that accepts dimension last.
    """

    def __init__(
        self,
        d_model: int,
        l_max: int,
        resid: bool = True,
        causal: bool = True,
        **kwargs,
    ):
        super().__init__()

        self.d_model = d_model
        self.kernel_size = l_max  # assumes all inputs are of shape (b, l_max, d)
        self.resid = resid
        self.causal = causal

        if self.causal:
            self.conv = nn.Conv1d(
                in_channels=d_model,
                out_channels=d_model,
                kernel_size=l_max,
                groups=d_model,
                padding=l_max - 1,
            )
        else:
            self.conv = nn.Conv1d(
                in_channels=d_model,
                out_channels=d_model,
                kernel_size=l_max,
                groups=d_model,
                padding=l_max // 2,
            )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (b, l, d) tensor
        Returns:
            y: (b, l, d) tensor
        """
        seq_n = x.size(1)
        y = self.conv(x.transpose(1, 2))[..., :seq_n].transpose(1, 2)
        if self.resid:
            y = y + x
        return y


class LongConvolution(nn.Module):
    """
    LongConvolution applies a convolution operation on the input tensor using a fixed
    filter of length l_max.
    The filter is learned during training and is applied using FFT convolution.
    Args:
        d_model (int): The number of expected features in the input and output.
        l_max (int): The maximum sequence length.
    Returns:
        y: (b, l, d) tensor
    """

    def __init__(
        self,
        d_model: int,
        l_max: int,
        resid: bool = True,
        causal: bool = True,
        **kwargs,
    ):
        """
        Initializes the LongConvolution module.
        Args:
            d_model (int): The number of expected features in the input and output.
            l_max (int): The maximum sequence length.
        """
        super().__init__()
        self.d_model = d_model
        self.resid = resid
        self.causal = causal

        # Define filter
        self.filter = nn.Parameter(torch.zeros(self.d_model, l_max), requires_grad=True)
        self.filter.data[:, 0] = 1

    def forward(self, x: torch.Tensor, *args, **kwargs):
        """
        Applies the LongConvolution operation on the input tensor.
        Args:
            x: (b, l, d) tensor
        Returns:
            y: (b, l, d) tensor
        """
        x = x.transpose(1, 2)
        y = fft_conv(
            x,
            self.filter,
            dropout_mask=None,
            gelu=False,
            resid=self.resid,
            causal=self.causal,
        )
        y = y.transpose(1, 2)
        return y.to(dtype=x.dtype)


class PositionalEmbedding(nn.Module):
    def __init__(self, emb_dim: int, seq_len: int, **kwargs):
        """Complex exponential positional embeddings for implicit long convolution filters."""
        super().__init__()

        self.seq_len = seq_len
        # The time embedding fed to the filteres is normalized so that t_f = 1
        t = torch.linspace(0, 1, self.seq_len)[None, :, None]  # 1, L, 1

        if emb_dim > 1:
            bands = (emb_dim - 1) // 2
        # To compute the right embeddings we use the "proper" linspace
        t_rescaled = torch.linspace(0, seq_len - 1, seq_len)[None, :, None]
        w = 2 * math.pi * t_rescaled / seq_len  # 1, L, 1

        f = torch.linspace(1e-4, bands - 1, bands)[None, None]
        z = torch.exp(-1j * f * w)
        z = torch.cat([t, z.real, z.imag], dim=-1)
        self.z = nn.Parameter(z, requires_grad=False)

    def forward(self, L):
        return self.z[:, :L]


class ImplicitLongConvolution(nn.Module):
    """
    Long convolution with implicit filter parameterized by an MLP.

    Args:
        d_model (int): The number of expected features in the input and output.
        l_max (int): The maximum sequence length.
        d_emb (int, optional): The dimension of the positional embeddings. Must be odd and greater or equal to 3 (time, sine and cosine). Defaults to 3.
        d_hidden (int, optional): The number of features in the hidden layer of the MLP. Defaults to 16.

    Attributes:
        pos_emb (PositionalEmbedding): The positional embedding layer.
        mlp (nn.Sequential): The MLP that parameterizes the implicit filter.

    """

    def __init__(
        self,
        d_model: int,
        l_max: int,
        d_emb: int = 3,
        d_hidden: int = 16,
        resid: bool = True,
        causal: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_emb = d_emb
        self.resid = resid
        self.causal = causal

        assert d_emb % 2 != 0 and d_emb >= 3, "d_emb must be odd and greater or equal to 3 (time, sine and cosine)"
        self.pos_emb = PositionalEmbedding(d_emb, l_max)

        # final linear layer
        self.mlp = nn.Sequential(
            nn.Linear(d_emb, d_hidden),
            torch.nn.ReLU(),
            nn.Linear(d_hidden, d_model),
        )

    def filter(self, ll: int, *args, **kwargs):
        k = self.mlp(self.pos_emb(ll))

        return k.transpose(1, 2)

    def forward(self, x: torch.Tensor, *args, **kwargs):
        """
        Args:
            x: (b, l, d) tensor
        Returns:
            y: (b, l, d) tensor
        """
        x = x.transpose(1, 2)
        k = self.filter(x.shape[-1])
        y = fft_conv(x, k, dropout_mask=None, gelu=False, resid=self.resid, causal=self.causal)

        y = y.transpose(1, 2)
        return y.to(dtype=x.dtype)


class BaseConv(nn.Module):
    def __init__(
        self,
        d_model: int,
        l_max: int,
        kernel_size: Union[int, List[int]] = -1,
        layer_idx: int = None,
        conv_type: str = "short",  # short, long, implicit
        resid: bool = True,
        causal: bool = True,
        **kwargs,
    ):
        super().__init__()

        self.d_model = d_model
        self.l_max = l_max
        self.layer_idx = layer_idx
        self.resid = resid
        self.conv_type = conv_type
        self.causal = causal

        self.projection = nn.Linear(self.d_model, self.d_model)
        self.in_proj = nn.Linear(self.d_model, self.d_model)
        self.out_proj = nn.Linear(self.d_model, self.d_model)

        # (Main, resid)
        # init = (1, 0)
        if self.resid:
            self.branch_weights = nn.Parameter(torch.zeros(2), requires_grad=True)
            self.branch_weights.data[0] = 1

        # support for different kernel sizes per layer
        if isinstance(kernel_size, List):
            if layer_idx is None or layer_idx >= len(kernel_size):
                raise ValueError(
                    "kernel_size must be an int or a list of ints with length equal to the number of layers"
                )
            kernel_size = kernel_size[layer_idx]

        # prepare convolution
        if kernel_size == -1:
            if conv_type == "implicit":
                conv = ImplicitLongConvolution
            elif conv_type == "long":
                conv = LongConvolution
            elif conv_type == "short":
                conv = ShortConvolution
            print(f"Conv type = {conv_type}")
            self.conv = conv(d_model, l_max=l_max, resid=self.resid, causal=self.causal)
        else:
            self.conv = ShortConvolution(d_model, kernel_size=kernel_size, resid=self.resid, causal=self.causal)

    def forward(self, u, *args, **kwargs):
        """
        Args:
            u: (b, l, d) tensor
        Returns:
            y: (b, l, d) tensor
        """
        u_conv = self.conv(self.in_proj(u))
        u_proj = self.projection(u)
        y = self.out_proj(u_conv * u_proj)
        if self.resid:
            return self.branch_weights[0] * y + self.branch_weights[1] * u
        else:
            return y


class BaseConvLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.base_conv = BaseConv(
            d_model=config.n_embd,
            l_max=config.block_size,
            kernel_size=-1,
            conv_type=config.conv_type,
            resid=config.use_resid,
            causal=config.causal,
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.base_conv(x)
        x = self.dropout(x)
        return x
