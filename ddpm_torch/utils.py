import math
import random
import torch
from torchvision.utils import make_grid
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams["figure.dpi"] = 144


def dict2str(d):
    out_str = []
    for k, v in d.items():
        out_str.append(str(k))
        if isinstance(v, (list, tuple)):
            v = "_".join(list(map(str, v)))
        elif isinstance(v, float):
            v = f"{v:.0e}"
        elif isinstance(v, dict):
            v = dict2str(v)
        out_str.append(str(v))
    out_str = "_".join(out_str)
    return out_str


def save_image(x, path, nrow=8, normalize=True, value_range=(-1., 1.)):
    img = make_grid(x, nrow=nrow, normalize=normalize, value_range=value_range)
    img = img.permute(1, 2, 0)
    _ = plt.imsave(path, img.numpy())


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_param(name, configs_1, configs_2):
    def get(obj, name):
        if hasattr(obj, "__getitem__"):
            return obj[name]
        elif hasattr(obj, "__getattribute__"):
            return getattr(obj, name)
        else:
            NotImplementedError("Not supported!")
    try:
        param = get(configs_1, name)
    except KeyError:
        param = get(configs_2, name)
    return param


def infer_range(dataset, precision=2):
    p = precision
    # infer proper x,y axes limits for evaluation/plotting
    xlim = np.array([-np.inf, np.inf])
    ylim = np.array([-np.inf, np.inf])
    _approx_clip = lambda x, y, z: np.clip([
        math.floor(p*x), math.ceil(p*y)], *z)
    for bch in dataset:
        xlim = _approx_clip(bch[:, 0].min(), bch[:, 0].max(), xlim)
        ylim = _approx_clip(bch[:, 1].min(), bch[:, 1].max(), ylim)
    return xlim / p, ylim / p


def save_scatterplot(fpath, x, y=None, xlim=None, ylim=None):
    if hasattr(x, "ndim"):
        x, y = split_squeeze(x) if x.ndim == 2 else (np.arange(len(x)), x)
    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, s=0.5, alpha=0.7)

    # set axes limits
    if xlim is not None:
        plt.xlim(*xlim)
    if ylim is not None:
        plt.ylim(*ylim)

    plt.tight_layout()
    plt.savefig(fpath)
    plt.close()  # close current figure before exit


def split_squeeze(data):
    x, y = np.split(data, 2, axis=1)
    x, y = x.squeeze(1), y.squeeze(1)
    return x, y