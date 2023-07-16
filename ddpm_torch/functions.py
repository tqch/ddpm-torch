import math
import numpy as np
import torch
import torch.nn.functional as F
from typing import Union, Tuple

DEFAULT_DTYPE = torch.float32


@torch.jit.script
def get_timestep_embedding(timesteps, embed_dim: int, dtype: torch.dtype = DEFAULT_DTYPE):
    """
    Adapted from fairseq/fairseq/modules/sinusoidal_positional_embedding.py
    The implementation is slightly different from the decription in Section 3.5 of [1]
    [1] Vaswani, Ashish, et al. "Attention is all you need."
     Advances in neural information processing systems 30 (2017).
    """
    half_dim = embed_dim // 2
    embed = math.log(10000) / (half_dim - 1)
    embed = torch.exp(-torch.arange(half_dim, dtype=dtype, device=timesteps.device) * embed)
    embed = torch.outer(timesteps.ravel().to(dtype), embed)
    embed = torch.cat([torch.sin(embed), torch.cos(embed)], dim=1)
    if embed_dim % 2 == 1:
        embed = F.pad(embed, [0, 1])  # padding the last dimension
    assert embed.dtype == dtype
    return embed


@torch.jit.script
def normal_kl(mean1, logvar1, mean2, logvar2):
    diff_logvar = logvar1 - logvar2
    kl = (-1.0 - diff_logvar).add(
        (mean1 - mean2).pow(2) * torch.exp(-logvar2)).add(
        torch.exp(diff_logvar)).mul(0.5)
    return kl


@torch.jit.script
def approx_std_normal_cdf(x):
    """
    Reference:
    Page, E. “Approximations to the Cumulative Normal Function and Its Inverse for Use on a Pocket Calculator.”
     Applied Statistics 26.1 (1977): 75–76. Web.
    """
    return 0.5 * (1. + torch.tanh(math.sqrt(2. / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


@torch.jit.script
def discretized_gaussian_loglik(
        x, means, log_scale, precision: float = 1./255,
        cutoff: Union[float, Tuple[float, float]] = (-0.999, 0.999), tol: float = 1e-12):
    if isinstance(cutoff, float):
        cutoff = (-cutoff, cutoff)
    # Assumes data is integers [0, 255] rescaled to [-1, 1]
    x_centered = x - means
    inv_stdv = torch.exp(-log_scale)
    upper = inv_stdv * (x_centered + precision)
    cdf_upper = torch.where(
        x > cutoff[1], torch.as_tensor(1, dtype=torch.float32, device=x.device), approx_std_normal_cdf(upper))
    lower = inv_stdv * (x_centered - precision)
    cdf_lower = torch.where(
        x < cutoff[0], torch.as_tensor(0, dtype=torch.float32, device=x.device), approx_std_normal_cdf(lower))
    log_probs = torch.log(torch.clamp(cdf_upper - cdf_lower - tol, min=0).add(tol))
    return log_probs


@torch.jit.script
def continuous_gaussian_loglik(x, mean, logvar):
    x_centered = x - mean
    inv_var = torch.exp(-logvar)
    log_probs = x_centered.pow(2) * inv_var + math.log(2 * math.pi) + logvar
    return log_probs.mul(0.5).neg()


def discrete_klv2d(hist1, hist2, eps=1e-9):
    """
    compute the discretized (empirical) Kullback-Leibler divergence between P_data1 and P_data2
    """
    return np.sum(hist2 * (np.log(hist2 + eps) - np.log(hist1 + eps)))


def hist2d(data, bins, value_range=None):
    """
    compute the 2d histogram matrix for a set of data points
    """
    if bins == "auto":
        bins = math.floor(math.sqrt(len(data) // 10))
    if value_range is not None:
        if isinstance(value_range, (int, float)):
            value_range = ((-value_range, value_range), ) * 2
        if hasattr(value_range, "__iter__"):
            if not hasattr(next(iter(value_range)), "__iter__"):
                value_range = (value_range, ) * 2
    x, y = np.split(data, 2, axis=1)
    x, y = x.squeeze(1), y.squeeze(1)
    return np.histogram2d(x, y, bins=bins, range=value_range)[0]


def flat_mean(x, start_dim=1):
    reduce_dim = [i for i in range(start_dim, x.ndim)]
    return torch.mean(x, dim=reduce_dim)


def flat_sum(x, start_dim=1):
    reduce_dim = [i for i in range(start_dim, x.ndim)]
    return torch.sum(x, dim=reduce_dim)
