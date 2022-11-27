import os
import json
import math
import uuid
import time
import torch
from tqdm import tqdm
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from ddpm_torch import *
from ddim import DDIM, get_selection_schedule
from argparse import ArgumentParser
import torch.multiprocessing as mp
from multiprocessing.sharedctypes import Synchronized


def generate(rank, args, counter=0):
    is_main = rank == 0
    dataset = args.dataset
    in_channels = DATA_INFO[dataset]["channels"]
    image_res = DATA_INFO[dataset]["resolution"][0]
    input_shape = (in_channels, image_res, image_res)

    config_dir = args.config_dir
    with open(os.path.join(config_dir, dataset + ".json")) as f:
        configs = json.load(f)

    diffusion_kwargs = configs["diffusion"]
    beta_schedule = diffusion_kwargs.pop("beta_schedule")
    beta_start = diffusion_kwargs.pop("beta_start")
    beta_end = diffusion_kwargs.pop("beta_end")
    num_diffusion_timesteps = diffusion_kwargs.pop("timesteps")
    betas = get_beta_schedule(beta_schedule, beta_start, beta_end, num_diffusion_timesteps)

    use_ddim = args.use_ddim
    if use_ddim:
        diffusion_kwargs["model_var_type"] = "fixed-small"
        skip_schedule = args.skip_schedule
        eta = args.eta
        subseq_size = args.subseq_size
        subsequence = get_selection_schedule(skip_schedule, size=subseq_size, timesteps=num_diffusion_timesteps)
        diffusion = DDIM(betas, **diffusion_kwargs, eta=eta, subsequence=subsequence)
    else:
        diffusion = GaussianDiffusion(betas, **diffusion_kwargs)

    device = torch.device(f"cuda:{rank}" if args.num_gpus > 1 else args.device)
    block_size = configs["denoise"].pop("block_size", 1)
    model = UNet(out_channels=in_channels, **configs["denoise"])
    if block_size > 1:
        pre_transform = torch.nn.PixelUnshuffle(block_size)  # space-to-depth
        post_transform = torch.nn.PixelShuffle(block_size)  # depth-to-space
        model = ModelWrapper(model, pre_transform, post_transform)
    model.to(device)
    chkpt_dir = args.chkpt_dir
    chkpt_path = args.chkpt_path or os.path.join(chkpt_dir, f"ddpm_{dataset}.pt")
    folder_name = os.path.basename(chkpt_path)[:-3]  # truncated at file extension
    use_ema = args.use_ema
    if use_ema:
        state_dict = torch.load(chkpt_path, map_location=device)["ema"]["shadow"]
    else:
        state_dict = torch.load(chkpt_path, map_location=device)["model"]
    for k in list(state_dict.keys()):
        if k.startswith("module."):  # state_dict of DDP
            state_dict[k.split(".", maxsplit=1)[1]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    del state_dict
    model.eval()
    for p in model.parameters():
        if p.requires_grad:
            p.requires_grad_(False)

    folder_name = folder_name + args.suffix
    save_dir = os.path.join(args.save_dir, folder_name)
    if is_main and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    world_size = args.num_gpus or 1
    total_size = args.total_size // world_size
    batch_size = args.batch_size
    if world_size > 1:
        remainder = args.total_size % world_size
        num_eval_batches = math.ceil((total_size + 1) / batch_size) * remainder
        num_eval_batches += math.ceil(total_size / batch_size) * (world_size - remainder)
        if rank < args.total_size % args.num_gpus:
            total_size += 1
        num_local_eval_batches = math.ceil(total_size / batch_size)
    else:
        num_eval_batches = num_local_eval_batches = math.ceil(total_size / batch_size)
    max_count = num_eval_batches
    shape = (batch_size, ) + input_shape

    def save_image(arr):
        with Image.fromarray(arr, mode="RGB") as im:
            im.save(f"{save_dir}/{uuid.uuid4()}.png")

    if torch.backends.cudnn.is_available():  # noqa
        torch.backends.cudnn.benchmark = True  # noqa

    class DummyProgressBar:
        def update(self, n):
            pass

    pbar = tqdm(total=max_count) if is_main else DummyProgressBar()
    local_counter = 0

    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        for i in range(num_local_eval_batches):
            if i == num_local_eval_batches - 1:
                shape = (total_size - i * batch_size, 3, image_res, image_res)
            x = diffusion.p_sample(model, shape=shape, device=device, noise=torch.randn(shape, device=device)).cpu()
            x = (x * 127.5 + 127.5).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).numpy()
            pool.map(save_image, list(x))
            if isinstance(counter, Synchronized):
                with counter.get_lock():
                    counter.value += 1
                    if counter.value > local_counter:
                        pbar.update(counter.value - local_counter)
                        local_counter = counter.value
            else:
                pbar.update(1)

    if is_main and isinstance(counter, Synchronized):
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # release unused cached memory
        while local_counter < max_count:
            time.sleep(0.1)
            with counter.get_lock():
                if counter.value > local_counter:
                    pbar.update(counter.value - local_counter)
                    local_counter = counter.value


def main():
    parser = ArgumentParser()
    parser.add_argument("--dataset", choices=["mnist", "cifar10", "celeba", "celebahq"], default="cifar10")
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--total-size", default=50000, type=int)
    parser.add_argument("--config-dir", default="./configs", type=str)
    parser.add_argument("--chkpt-dir", default="./chkpts", type=str)
    parser.add_argument("--chkpt-path", default="", type=str)
    parser.add_argument("--save-dir", default="./images/eval", type=str)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--use-ddim", action="store_true")
    parser.add_argument("--eta", default=0., type=float)
    parser.add_argument("--skip-schedule", default="linear", type=str)
    parser.add_argument("--subseq-size", default=10, type=int)
    parser.add_argument("--suffix", default="", type=str)
    parser.add_argument("--max-workers", default=8, type=int)
    parser.add_argument("--num-gpus", default=1, type=int)

    args = parser.parse_args()

    world_size = args.num_gpus
    if world_size > 1:
        mp.set_start_method("spawn")
        counter = mp.Value("i", 0)
        mp.spawn(generate, args=(args, counter), nprocs=world_size)
    else:
        generate(0, args)


if __name__ == "__main__":
    main()
