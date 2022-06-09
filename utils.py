import random
import torch
from torchvision.utils import make_grid
import numpy as np
import matplotlib.pyplot as plt


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


def restart_from_chkpt(chkpt_path, model, optimizers, device=torch.device("cpu")):
    chkpt = torch.load(chkpt_path, map_location=device)
    model.load_state_dict(chkpt["model"])
    for k in optimizers.keys():
        optimizers[k].load_state_dict(chkpt[k])
    fid = chkpt["fid"]
    return fid, model, optimizers


class RunningStatistics:
    def __init__(self, **kwargs):
        self.count = 0
        self.stats = []
        for k, v in kwargs.items():
            self.stats.append((k, v or 0))
        self.stats = dict(self.stats)

    def reset(self):
        self.count = 0
        for k in self.stats:
            self.stats[k] = 0

    def update(self, n, **kwargs):
        self.count += n
        for k, v in kwargs.items():
            self.stats[k] = self.stats.get(k, 0) + v

    def extract(self):
        avg_stats = []
        for k, v in self.stats.items():
            avg_stats.append((k, v/self.count))
        return dict(avg_stats)

    def __repr__(self):
        out_str = "Count(s): {}\n"
        out_str += "Statistics:\n"
        for k in self.stats:
            out_str += f"\t{k} = {{{k}}}\n"  # double curly-bracket to escape
        return out_str.format(self.count, **self.stats)


def pair(x):
    return (x, x)


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
