import re
import os
import csv
import PIL
import torch
import numpy as np
from torchvision import transforms, datasets as tvds
from torch.utils.data import DataLoader, Subset, Sampler
from torch.utils.data.distributed import DistributedSampler
from collections import namedtuple

CSV = namedtuple("CSV", ["header", "index", "data"])
DATASET_DICT = dict()
DATASET_INFO = dict()


def register_dataset(cls):
    name = cls.__name__.lower()
    DATASET_DICT[name] = cls
    info = dict()
    for k, v in cls.__dict__.items():
        if re.match(r"__\w+__", k) is None and not callable(v):
            info[k] = v
    DATASET_INFO[name] = info
    return cls


@register_dataset
class MNIST(tvds.MNIST):
    resolution = (32, 32)
    channels = 1
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_size = 60000
    test_size = 10000


@register_dataset
class CIFAR10(tvds.CIFAR10):
    resolution = (32, 32)
    channels = 3
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    _transform = transforms.PILToTensor()
    train_size = 50000
    test_size = 10000


def crop_celeba(img):
    return transforms.functional.crop(img, top=40, left=15, height=148, width=148)  # noqa


@register_dataset
class CelebA(tvds.VisionDataset):
    """
    Large-scale CelebFaces Attributes (CelebA) Dataset <https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html>
    """
    base_folder = "celeba"
    resolution = (64, 64)
    channels = 3
    transform = transforms.Compose([
        crop_celeba,
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    _transform = transforms.Compose([
        crop_celeba,
        transforms.Resize((64, 64)),
        transforms.PILToTensor()
    ])
    all_size = 202599
    train_size = 162770
    test_size = 19962
    valid_size = 19867

    def __init__(
            self,
            root,
            split,
            transform=None
    ):
        super().__init__(root, transform=transform or self._transform)
        self.split = split
        split_map = {
            "train": 0,
            "valid": 1,
            "test": 2,
            "all": None,
        }
        split_ = split_map[split.lower()]
        splits = self._load_csv("list_eval_partition.txt")
        mask = slice(None) if split_ is None else (splits.data == split_).squeeze()
        if mask == slice(None):  # if split == "all"
            self.filename = splits.index
        else:
            self.filename = [splits.index[i] for i in torch.squeeze(torch.nonzero(mask))]

    def _load_csv(
            self,
            filename,
            header=None,
    ):
        with open(os.path.join(self.root, self.base_folder, filename)) as csv_file:
            data = list(csv.reader(csv_file, delimiter=" ", skipinitialspace=True))

        if header is not None:
            headers = data[header]
            data = data[header + 1:]
        else:
            headers = []

        indices = [row[0] for row in data]
        data = [row[1:] for row in data]
        data_int = [list(map(int, i)) for i in data]

        return CSV(headers, indices, torch.as_tensor(data_int))

    def __getitem__(self, index):
        im = PIL.Image.open(os.path.join(  # noqa
            self.root, self.base_folder, "img_align_celeba", self.filename[index]))

        if self.transform is not None:
            im = self.transform(im)

        return im

    def __len__(self):
        return len(self.filename)

    def extra_repr(self):
        lines = ["Split: {split}", ]
        return "\n".join(lines).format(**self.__dict__)


@register_dataset
class CelebAHQ(tvds.VisionDataset):
    """
    High-Quality version of the CELEBA dataset, consisting of 30000 images in 1024 x 1024 resolution
    created by Karras et al. (2018) [1]
    [1] Karras, Tero, et al. "Progressive Growing of GANs for Improved Quality, Stability, and Variation." International Conference on Learning Representations. 2018.
    """  # noqa
    base_folder = "celeba_hq"
    resolution = (256, 256)
    channels = 3
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    _transform = transforms.PILToTensor()
    all_size = 30000

    def __init__(
            self,
            root,
            transform=None
    ):
        super().__init__(root, transform=transform or self._transform)
        self.filename = sorted([
            fname
            for fname in os.listdir(os.path.join(root, self.base_folder, "img_celeba_hq"))
            if fname.endswith(".png")
        ], key=lambda name: int(name[:-4].zfill(5)))
        np.random.RandomState(123).shuffle(self.filename)  # legacy order used by ProGAN

    def __getitem__(self, index):
        im = PIL.Image.open(os.path.join(  # noqa
            self.root, self.base_folder, "img_celeba_hq", self.filename[index]))

        if self.transform is not None:
            im = self.transform(im)

        return im

    def __len__(self):
        return len(self.filename)


ROOT = os.path.expanduser("~/datasets")


def train_val_split(dataset, val_size, random_seed=None):
    train_size = DATASET_INFO[dataset]["train_size"]
    if random_seed is not None:
        np.random.seed(random_seed)
    train_inds = np.arange(train_size)
    np.random.shuffle(train_inds)
    val_size = int(train_size * val_size)
    val_inds, train_inds = train_inds[:val_size], train_inds[val_size:]
    return train_inds, val_inds


class SubsetSequentialSampler(Sampler):
    def __init__(self, indices):  # noqa
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def get_dataloader(
        dataset,
        batch_size,
        split,
        val_size=0.1,
        random_seed=None,
        root=ROOT,
        pin_memory=False,
        drop_last=False,
        num_workers=0,
        distributed=False
):
    assert isinstance(val_size, float) and 0 <= val_size < 1

    name, dataset = dataset, DATASET_DICT[dataset]
    transform = dataset.transform
    if distributed:
        batch_size = batch_size // int(os.environ.get("WORLD_SIZE", "1"))
    data_kwargs = {"root": root, "transform": transform}
    if name == "celeba":
        data_kwargs["split"] = split
    elif name in {"mnist", "cifar10"}:
        data_kwargs["download"] = False
        data_kwargs["train"] = split != "test"
    dataset = dataset(**data_kwargs)

    if data_kwargs.get("train", False) and val_size > 0.:
        train_inds, val_inds = train_val_split(name, val_size, random_seed)
        dataset = Subset(dataset, {"train": train_inds, "valid": val_inds}[split])

    dataloader_configs = {
        "batch_size": batch_size,
        "pin_memory": pin_memory,
        "drop_last": drop_last,
        "num_workers": num_workers
    }
    dataloader_configs["sampler"] = sampler = DistributedSampler(
        dataset, shuffle=True, seed=random_seed, drop_last=drop_last) if distributed else None
    dataloader_configs["shuffle"] = (sampler is None) if split in {"train", "all"} else False
    dataloader = DataLoader(dataset, **dataloader_configs)
    return dataloader, sampler


if __name__ == "__main__":
    try:
        from .utils import dict2str
    except ImportError:
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).resolve().parents[0]))
        from utils import dict2str

    print(dict2str(DATASET_INFO, compact=False))
