if __name__ == "__main__":
    import os
    import math
    import torch
    import numpy as np
    from PIL import Image
    from torch.utils.data import Dataset, Subset, DataLoader
    from torchvision import transforms
    from ddpm_torch import *
    from ddpm_torch.metrics import *
    from tqdm import tqdm
    from functools import partial
    from copy import deepcopy
    from argparse import ArgumentParser

    parser = ArgumentParser()

    parser.add_argument("--root", default="~/datasets", type=str)
    parser.add_argument("--dataset", choices=["mnist", "cifar10", "celeba", "celebahq"], default="cifar10")
    parser.add_argument("--eval-batch-size", default=512, type=int)
    parser.add_argument("--eval-total-size", default=50000, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--nhood-size", default=3, type=int)
    parser.add_argument("--row-batch-size", default=10000, type=int)
    parser.add_argument("--col-batch-size", default=10000, type=int)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--eval-dir", default="./images/eval")
    parser.add_argument("--precomputed-dir", default="./precomputed", type=str)
    parser.add_argument("--metrics", nargs="+", default=["fid", "pr"], type=str)
    parser.add_argument("--seed", default=1234, type=int)
    parser.add_argument("--folder-name", default="", type=str)

    args = parser.parse_args()

    root = os.path.expanduser(args.root)
    dataset = args.dataset
    print(f"Dataset: {dataset}")

    img_dir = eval_dir = args.eval_dir
    folder_name = args.folder_name
    if folder_name:
        img_dir = os.path.join(img_dir, folder_name)
    device = torch.device(args.device)

    args = parser.parse_args()

    precomputed_dir = args.precomputed_dir
    eval_batch_size = args.eval_batch_size
    eval_total_size = args.eval_total_size
    num_workers = args.num_workers


    class ImageFolder(Dataset):
        def __init__(self, img_dir, transform=transforms.PILToTensor()):
            self.img_dir = img_dir
            self.img_list = [
                f for f in os.listdir(img_dir)
                if f.split(".")[-1] in {"jpg", "jpeg", "png", "bmp", "webp", "tiff"}]
            self.transform = transform

        def __getitem__(self, idx):
            with Image.open(os.path.join(self.img_dir, self.img_list[idx])) as im:
                return self.transform(im)

        def __len__(self):
            return len(self.img_list)

    seed_all(args.seed)

    imagefolder = ImageFolder(img_dir)
    if len(imagefolder) > eval_total_size:
        inds = torch.as_tensor(np.random.choice(len(imagefolder), size=eval_total_size, replace=False))
        imagefolder = Subset(imagefolder, indices=inds)
    imageloader = DataLoader(
        imagefolder, batch_size=eval_batch_size, shuffle=False,
        num_workers=num_workers, drop_last=False, pin_memory=True)


    def eval_fid():
        istats = InceptionStatistics(device=device, input_transform=lambda im: (im-127.5) / 127.5)
        true_mean, true_var = get_precomputed(dataset, download_dir=precomputed_dir)
        istats.reset()
        for x in tqdm(imageloader, desc="Computing Inception statistics"):
            istats(x.to(device))
        gen_mean, gen_var = istats.get_statistics()
        fid = calc_fd(gen_mean, gen_var, true_mean, true_var)
        return fid

    row_batch_size = args.row_batch_size
    col_batch_size = args.col_batch_size
    nhood_size = args.nhood_size


    def eval_pr():
        decimal_places = math.ceil(math.log(eval_total_size, 10))
        str_fmt = f".{decimal_places}f"
        _ManifoldBuilder = partial(
            ManifoldBuilder, extr_batch_size=eval_batch_size, max_sample_size=eval_total_size,
            row_batch_size=row_batch_size, col_batch_size=col_batch_size, nhood_size=nhood_size,
            num_workers=num_workers, device=device)
        manifold_path = os.path.join(precomputed_dir, f"pr_manifold_{dataset}.pt")
        if not os.path.exists(manifold_path):
            dataset_kwargs = {
                "celeba": {"split": "all"},
            }.get(dataset, {"train": True})
            transform = DATASET_DICT[dataset].__dict__.get(
                "_transform", DATASET_DICT[dataset].__dict__.get("transform", None))
            manifold_builder = _ManifoldBuilder(
                data=DATASET_DICT[dataset](root=root, transform=transform, **dataset_kwargs))
            manifold_builder.save(manifold_path)
            true_manifold = deepcopy(manifold_builder.manifold)
            del manifold_builder
        else:
            true_manifold = torch.load(manifold_path)
        gen_manifold = deepcopy(_ManifoldBuilder(data=imagefolder).manifold)

        precision, recall = calc_pr(
            gen_manifold, true_manifold,
            row_batch_size=row_batch_size, col_batch_size=col_batch_size, device=device)
        return f"{precision:{str_fmt}}/{recall:{str_fmt}}"

    def warning(msg):
        def print_warning():
            print(msg)
        return print_warning

    for metric in set(args.metrics):
        result = {"fid": eval_fid, "pr": eval_pr}.get(metric, warning("Unsupported metric passed! Ignore."))()
        print(f"{metric.upper()}: {result}")
