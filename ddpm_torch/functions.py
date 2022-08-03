import math
import torch
import torch.nn.functional as F
import numpy as np


DEFAULT_DTYPE = torch.float32


def get_timestep_embedding(timesteps, embed_dim):
    half_dim = embed_dim // 2
    embed = math.log(10000) / (half_dim - 1)
    embed = torch.exp(-torch.arange(half_dim, dtype=DEFAULT_DTYPE) * embed)
    embed = torch.outer(timesteps.to(DEFAULT_DTYPE), embed)
    embed = torch.cat([torch.sin(embed), torch.cos(embed)], dim=1)
    if embed_dim % 2 == 1:
        embed = F.pad(embed, [0, 1])  # padding the last dimension
    assert embed.dtype == DEFAULT_DTYPE
    return embed


@torch.jit.script
def normal_kl(mean1, logvar1, mean2, logvar2):
    diff_logvar = logvar1 - logvar2
    kl = (-1.0 - diff_logvar).add(
        (mean1 - mean2).pow(2) * torch.exp(-logvar2)).add(
        torch.exp(diff_logvar)).mul(0.5)
    return kl


def approx_std_normal_cdf(x):
    #  E. Page (1977)
    # "Approximating to the Cumulative Normal function and its Inverse for use on a Pocket Calculator"
    return 0.5 * (1. + torch.tanh(math.sqrt(2. / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def discretized_gaussian_loglik(x, means, log_scale):
    # Assumes data is integers [0, 255] rescaled to [-1, 1]
    x_centered = x - means
    inv_stdv = torch.exp(-log_scale)
    upper = inv_stdv * (x_centered + 1./255)
    cdf_upper = torch.where(
        x > 0.999, torch.as_tensor(1, dtype=torch.float32, device=x.device), approx_std_normal_cdf(upper))
    lower = inv_stdv * (x_centered - 1./255)
    cdf_lower = torch.where(
        x < -0.999, torch.as_tensor(0, dtype=torch.float32, device=x.device), approx_std_normal_cdf(lower))
    log_probs = torch.log(torch.clamp(cdf_upper - cdf_lower - 1e-12, min=0).add(1e-12))
    return log_probs


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

