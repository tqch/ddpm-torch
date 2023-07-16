import math
import numpy as np
import torch
from sklearn.datasets import make_swiss_roll
from torch.utils.data import Dataset

__all__ = ["Gaussian8", "Gaussian25", "SwissRoll", "DataStreamer"]


class ToyDataset(Dataset):
    def __init__(self, size: int, stdev: float, random_state: int = None):
        self.size = size
        self.noise = stdev
        self.random_state = random_state
        self.stdev = self._calc_stdev()
        self.data = self._sample()
        
    def _calc_stdev(self):
        pass

    def _sample(self):
        pass

    def resample(self):
        self.data = self._sample()

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx])


class Gaussian8(ToyDataset):
    scale = 2
    modes = [
        (math.cos(0.25 * t * math.pi), math.sin(0.25 * t * math.pi))
        for t in range(8)
    ]  # scale x (8 roots of z^8 = 1)

    def __init__(self, size, stdev=0.02, random_state=1234):
        self.modes = self.scale * np.array(self.modes, dtype=np.float32)
        super(Gaussian8, self).__init__(size, stdev, random_state)
    
    def _calc_stdev(self):
        # total variance = expected conditional variance + variance of conditional expectation
        return math.sqrt(self.noise ** 2 + (self.scale ** 2) * 0.5)  # x-y symmetric; around 1.414
    
    def _sample(self):
        rng = np.random.default_rng(seed=self.random_state)
        data = self.noise * rng.standard_normal((self.size, 2), dtype=np.float32)
        data += np.array(self.modes)[
            np.random.choice(np.arange(8), size=self.size, replace=True)]
        data /= self.stdev
        return data


class Gaussian25(ToyDataset):
    scale = 2
    modes = [(i, j) for i in range(-2, 3) for j in range(-2, 3)]

    def __init__(self, size, stdev=0.05, random_state=1234):
        self.modes = self.scale * np.array(self.modes, dtype=np.float32)
        super(Gaussian25, self).__init__(size, stdev, random_state)

    def _calc_stdev(self):
        # x-y symmetric; around 2.828
        return math.sqrt(self.noise ** 2 + (self.scale ** 2) * 2.)  

    def _sample(self):
        rng = np.random.default_rng(self.random_state)
        data = self.noise * rng.standard_normal((self.size, 2), dtype=np.float32)
        data += np.array(self.modes)[np.arange(self.size) % 25]
        data /= self.stdev
        return data


class SwissRoll(ToyDataset):
    """
    source: https://homepages.ecs.vuw.ac.nz/~marslast/Code/Ch6/lle.py
    def swissroll():
        # Make the swiss roll dataset
        N = 1000
        noise = 0.05

        t = 3*np.pi/2 * (1 + 2*np.random.rand(1,N))
        h = 21 * np.random.rand(1,N)
        data = np.concatenate((t*np.cos(t),h,t*np.sin(t))) + noise*np.random.randn(3,N)
        return np.transpose(data), np.squeeze(t)

    The covariate standard deviation of x,y without noise
    E[x] = 2 and var(x) = (39/8)*pi^2 - 17/4
    E[y] = 2/(3*pi) and var(y) = (39/8)*pi^2 - 15/4
    """

    def __init__(self, size, stdev=0.25, random_state=1234):
        super(SwissRoll, self).__init__(size, stdev, random_state)

    def _calc_stdev(self):
        # calculate the stdev's for the data
        stdev = np.empty((1, 2))
        stdev.fill(39 * math.pi ** 2 / 8 - 4)
        stdev += np.array([[-1, 1]]) * 0.25 + self.noise ** 2
        stdev = np.sqrt(stdev)
        return stdev

    def _sample(self):
        data = make_swiss_roll(
            self.size, noise=self.noise,
            random_state=self.random_state)[0][:, [0, 2]].astype(np.float32)
        data /= self.stdev
        return data


class DataStreamer:

    def __init__(self, dataset: ToyDataset, batch_size: int, num_batches: int, resample: bool = False):
        
        dataset = self.dataset_map(dataset)
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.resample = resample
        self.dataset = dataset(batch_size * num_batches, random_state=None)

    def __iter__(self):
        cnt = 0
        while True:
            start = cnt * self.batch_size
            end = start + self.batch_size
            yield torch.from_numpy(self.dataset.data[start:end])
            cnt += 1
            if cnt >= self.num_batches:
                break
        if self.resample:
            self.dataset.resample()

    def __len__(self):
        return self.num_batches
        
    @staticmethod
    def dataset_map(dataset):
        return {
            "gaussian8": Gaussian8,
            "gaussian25": Gaussian25,
            "swissroll": SwissRoll
        }.get(dataset, None)


if __name__ == "__main__":
    import os
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader

    mpl.rcParams["figure.dpi"] = 144

    fig_dir = "./figs"
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    size = 100000

    DATASET = {
            "gaussian8": Gaussian8,
            "gaussian25": Gaussian25,
            "swissroll": SwissRoll
    }

    for name, dataset in DATASET.items():
        data = dataset(size)
        plt.figure(figsize=(6, 6))
        plt.scatter(*np.hsplit(data.data, 2), s=0.5, alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f"{name}.jpg"))
        dataloader = DataLoader(data)
        x = next(iter(dataloader))
