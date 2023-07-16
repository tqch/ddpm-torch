import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import random
import torch

mpl.rcParams["figure.dpi"] = 144


def dict2str(d, level=0, compact=True):
    out_str = []
    if compact:
        indents, newline, colon, comma = "." * level, "", "(", ")+"
        brackets = "", ""
    else:
        indents, newline, colon, comma = "  " * level, "\n", ": ", ","
        brackets = "{", "}"
    for i, (k, v) in enumerate(d.items()):
        line = indents + str(k) + colon
        if isinstance(v, str):
            line += v
        elif isinstance(v, float):
            line += f"{v:.3e}"
        elif isinstance(v, dict):
            line += brackets[0] + newline + dict2str(v, level + 1, compact=compact)
            line += indents + brackets[1]
        else:
            if compact and isinstance(v, (list, tuple)):
                line += "_".join(list(map(str, v)))
            else:
                line += str(v)
        if i != len(d) - 1:
            line += comma
        line += newline
        out_str.append(line)
    return "".join(out_str)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_param(param, obj_1, obj_2):
    def get(obj, attr):
        if hasattr(obj, "__getitem__"):
            return obj[attr]
        elif hasattr(obj, "__getattribute__"):
            return getattr(obj, attr)
        else:
            NotImplementedError("Not supported!")
    try:
        param = get(obj_1, param)
    except (KeyError, AttributeError):
        param = get(obj_2, param)
    return param


def infer_range(dataset):
    # infer proper x,y axes limits for evaluation/plotting
    xlim = (dataset.data[:, 0].min(), dataset.data[:, 0].max())
    ylim = (dataset.data[:, 1].min(), dataset.data[:, 1].max())
    x_range = xlim[1] - xlim[0]
    y_range = ylim[1] - ylim[0]
    xlim = (xlim[0] - 0.05 * x_range, xlim[1] + 0.05 * x_range)
    ylim = (ylim[0] - 0.05 * y_range, ylim[1] + 0.05 * y_range)
    return xlim, ylim


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


class ConfigDict(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __getattr__(self, name):
        return self.get(name, None)
