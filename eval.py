if __name__ == "__main__":
    import math
    import numpy as np
    import os
    import torch
    from PIL import Image
    from argparse import ArgumentParser
    from copy import deepcopy
    from ddpm_torch import *
    from ddpm_torch.metrics import *
    from functools import partial
    from torch.utils.data import Dataset, Subset, DataLoader
    from torchvision import transforms
    from tqdm import tqdm

    parser = ArgumentParser()

    parser.add_argument("--root", default="~/datasets", type=str)
    parser.add_argument("--dataset", choices=DATASET_DICT.keys(), default="cifar10")
    parser.add_argument("--eval-batch-size", default=512, type=int)
    parser.add_argument("--eval-total-size", default=50000, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--nhood-size", default=3, type=int)
    parser.add_argument("--row-batch-size", default=10000, type=int)
    parser.add_argument("--col-batch-size", default=10000, type=int)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--precomputed-dir", default="./precomputed", type=str)
    parser.add_argument("--metrics", nargs="+", default=["fid", "pr"], type=str)
    parser.add_argument("--seed", default=1234, type=int)
    parser.add_argument("--sample-folder", default="", type=str)
    parser.add_argument("--num-gpus", default=1, type=int)

    args = parser.parse_args()

    root = os.path.expanduser(args.root)
    dataset = args.dataset
    print(f"Dataset: {dataset}")

    folder_name = os.path.basename(args.sample_folder.rstrip(r"\/"))
    if args.num_gpus > 1:
        assert torch.cuda.is_available() and torch.cuda.device_count() >= args.num_gpus
        model_device = [f"cuda:{i}" for i in range(args.num_gpus)]
        input_device = "cpu"  # nn.DataParallel is input device agnostic
        op_device = model_device[0]
    else:
        op_device = input_device = model_device = torch.device(args.device)

    args = parser.parse_args()

    precomputed_dir = args.precomputed_dir
    eval_batch_size = args.eval_batch_size
    eval_total_size = args.eval_total_size
    num_workers = args.num_workers


    class ImageFolder(Dataset):
        def __init__(self, img_dir, transform=transforms.PILToTensor()):
            self.img_dir = img_dir
            self.img_list = [
                img for img in os.listdir(img_dir)
                if img.split(".")[-1] in {"jpg", "jpeg", "png", "bmp", "webp", "tiff"}]
            self.transform = transform

        def __getitem__(self, idx):
            with Image.open(os.path.join(self.img_dir, self.img_list[idx])) as im:
                return self.transform(im)

        def __len__(self):
            return len(self.img_list)

    seed_all(args.seed)

    imagefolder = ImageFolder(args.sample_folder)
    if len(imagefolder) > eval_total_size:
        inds = torch.as_tensor(np.random.choice(len(imagefolder), size=eval_total_size, replace=False))
        imagefolder = Subset(imagefolder, indices=inds)
    imageloader = DataLoader(
        imagefolder, batch_size=eval_batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False)


    def eval_fid():
        istats = InceptionStatistics(device=model_device, input_transform=lambda im: (im-127.5) / 127.5)
        try:
            true_mean, true_var = get_precomputed(dataset, download_dir=precomputed_dir)
        except Exception:
            print("Precomputed statistics cannot be loaded! Computing from raw data...")
            dataloader = get_dataloader(
                dataset, batch_size=eval_batch_size, split="all", val_size=0., root=root,
                pin_memory=True, drop_last=False, num_workers=num_workers, raw=True)[0]
            for x in tqdm(dataloader):
                istats(x.to(input_device))
            true_mean, true_var = istats.get_statistics()
            np.savez(os.path.join(precomputed_dir, f"fid_stats_{dataset}.npz"), mu=true_mean, sigma=true_var)
        istats.reset()
        for x in tqdm(imageloader, desc="Computing Inception statistics"):
            istats(x.to(input_device))
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
            num_workers=num_workers, device=model_device)
        manifold_path = os.path.join(precomputed_dir, f"pr_manifold_{dataset}.pt")
        if not os.path.exists(manifold_path):
            manifold_builder = _ManifoldBuilder(
                data=DATASET_DICT[dataset](root=root, split="all", transform=None))
            manifold_builder.save(manifold_path)
            true_manifold = deepcopy(manifold_builder.manifold)
            del manifold_builder
        else:
            true_manifold = torch.load(manifold_path)
        gen_manifold = deepcopy(_ManifoldBuilder(data=imagefolder).manifold)

        precision, recall = calc_pr(
            gen_manifold, true_manifold,
            row_batch_size=row_batch_size, col_batch_size=col_batch_size, device=op_device)
        return f"{precision:{str_fmt}}/{recall:{str_fmt}}"

    def warning(msg):
        def print_warning():
            print(msg)
        return print_warning

    result_dict = {"folder_name": folder_name}
    with open(os.path.join(os.path.dirname(args.sample_folder.rstrip(r"\/")), "metrics.txt"), "a") as f:
        for metric in args.metrics:
            result = {"fid": eval_fid, "pr": eval_pr}.get(metric, warning("Unsupported metric passed! Ignore."))()
            print(f"{metric.upper()}: {result}")
            result_dict[metric] = result
        f.write(str(result_dict))
