import os, json
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datasets import get_dataloader, DATA_INFO
from datetime import datetime
from utils import dict2str, RunningStatistics, save_image, seed_all
from metrics.fid_score import InceptionStatistics, get_precomputed, fid
from diffusion import GaussianDiffusion, get_beta_schedule
from models import UNet
import matplotlib as mpl
mpl.rcParams["figure.dpi"] = 144

MODEL_LIST = {
    "unet": UNet
}


class DummyScheduler:
    def init(self): pass
    def step(self): pass
    def load_state_dict(self, state_dict): pass
    def state_dict(self): return None


class Trainer:
    def __init__(
            self,
            model,
            optimizer,
            diffusion,
            epochs,
            trainloader,
            scheduler=DummyScheduler(),
            grad_norm=1.0,
            shape=None,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            chkpt_intv=5,  # save a checkpoint every {chkpt_intv} epochs
            num_save_images=64
    ):

        self.model = model
        self.optimizer = optimizer
        self.diffusion = diffusion
        self.epochs = epochs
        self.start_epoch = 0
        self.trainloader = trainloader
        if shape is None:
            shape = next(iter(trainloader))[0].shape[1:]
        self.shape = shape
        self.scheduler = scheduler
        self.grad_norm = grad_norm
        self.device = device
        self.chkpt_intv = chkpt_intv
        self.num_save_images = num_save_images

        model.to(device)
        self.stats = RunningStatistics(loss=None)

    def loss(self, x):
        B = x.shape[0]
        T = self.diffusion.timesteps
        t = torch.randint(T - 1, size=(B, ), dtype=torch.int64)
        loss = self.diffusion.train_losses(self.model, x_0=x, t=t)
        assert loss.shape == (B, )
        return loss

    def step(self, x):
        B = x.shape[0]
        loss = self.loss(x).mean()
        self.optimizer.zero_grad()
        loss.backward()
        # gradient clipping by global norm
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_norm)
        self.optimizer.step()
        self.stats.update(B, loss=loss.item() * B)

    def train(self, evaluator=None, chkpt_path=None, image_dir=None):

        def sample_fn(noise):
            shape = noise.shape
            with torch.no_grad():
                sample = self.diffusion.p_sample(
                    denoise_fn=self.model, shape=shape, device=self.device, noise=noise)
            return sample
        image_idx = 0
        num_samples = self.num_save_images
        if num_samples:
            noise = torch.randn((num_samples,) + self.shape)  # fixed x_T for image generation
        for e in range(self.start_epoch, self.epochs):
            self.stats.reset()
            self.model.train()
            with tqdm(self.trainloader, desc=f"{e+1}/{self.epochs} epochs") as t:
                for i, (x, _) in enumerate(t):
                    self.step(x.to(self.device))
                    t.set_postfix(self.current_stats)
                    if i == len(self.trainloader) - 1:
                        self.model.eval()
                        if evaluator is not None:
                            eval_results = evaluator.eval(sample_fn)
                        else:
                            eval_results = dict()
                        results = dict()
                        results.update(self.current_stats)
                        results.update(eval_results)
                        t.set_postfix(results)
            if num_samples and image_dir:
                with torch.no_grad():
                    x = sample_fn(noise).cpu()
                    save_image(x, os.path.join(image_dir, f"{image_idx+1}.jpg"))
                    image_idx += 1
            # adjust learning rate every epoch before checkpoint
            scheduler.step()
            if (e+1) % self.chkpt_intv and chkpt_path:
                self.save_checkpoint(chkpt_path, epoch=e+1, **results)

    @property
    def current_stats(self):
        return self.stats.extract()

    def restart_from_chkpt(self, chkpt_path):
        chkpt = torch.load(chkpt_path, map_location=self.device)
        self.model.load_state_dict(chkpt["model"])
        self.optimizer.load_state_dict(chkpt["optimizer"])
        self.scheduler.load_state_dict(chkpt["scheduler"])
        self.start_epoch = chkpt["epoch"]

    def save_checkpoint(self, chkpt_path, **extra_info):
        chkpt = []
        for k, v in self.named_state_dicts():
            chkpt.append((k, v))
        for k, v in extra_info.items():
            chkpt.append((k, v))
        torch.save(dict(chkpt), chkpt_path)

    def named_state_dicts(self):
        for k in ["model", "optimizer", "scheduler"]:
            yield k, getattr(self, k).state_dict()


class Evaluator:
    def __init__(
            self,
            eval_batch_size=256,
            max_eval_count=10000,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ):
        # inception stats
        self.istats = InceptionStatistics(device=device)
        self.eval_batch_size = eval_batch_size
        self.max_eval_count = max_eval_count
        self.device = device
        self.target_mean, self.target_var = get_precomputed(dataset)

    def eval(self, sample_fn):
        self.istats.reset()
        with torch.no_grad():
            for _ in range(0, self.max_eval_count + self.eval_batch_size, self.eval_batch_size):
                with torch.no_grad():
                    x = sample_fn(self.eval_batch_size)
                self.istats(x.to(self.device))
        gen_mean, gen_var = self.istats.get_statistics()
        return {"fid": fid(gen_mean, self.target_mean, gen_var, self.target_var)}


if __name__ == "__main__":
    from argparse import ArgumentParser
    from utils import get_param
    from functools import partial

    parser = ArgumentParser()
    parser.add_argument("--model", choices=["unet", ], default="unet")
    parser.add_argument("--dataset", choices=["mnist", "cifar10", "celeba"], default="cifar10")
    parser.add_argument("--root", default="~/datasets", type=str)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--lr", default=0.0002, type=float)
    parser.add_argument("--beta1", default=0.5, type=float)
    parser.add_argument("--beta2", default=0.999, type=float)
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--timesteps", default=1000, type=int)
    parser.add_argument("--beta-schedule", choices=["quad", "linear", "warmup10", "warmup50", "jsd"], default="linear")
    parser.add_argument("--beta-start", default=0.0001, type=float)
    parser.add_argument("--beta-end", default=0.02, type=float)
    parser.add_argument("--model-mean-type", choices=["mean", "x_0", "eps"], default="eps", type=str)
    parser.add_argument("--model-var-type", choices=["learned", "fixed-small", "fixed-large"], default="fixed-large", type=str)
    parser.add_argument("--loss-type", choices=["kl", "mse"], default="mse", type=str)
    parser.add_argument("--task", choices=["generation", ], default="generation")
    parser.add_argument("--train-device", default="cuda:0", type=str)
    parser.add_argument("--eval-device", default="cuda:0", type=str)
    parser.add_argument("--latent-dim", default=128, type=int)
    parser.add_argument("--image-dir", default="./images", type=str)
    parser.add_argument("--config-dir", default="./configs", type=str)
    parser.add_argument("--chkpt-dir", default="./chkpts", type=str)
    parser.add_argument("--log-dir", default="./logs", type=str)
    parser.add_argument("--seed", default=1234, type=int)
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--eval", action="store_true")

    args = parser.parse_args()

    root = os.path.expanduser(args.root)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = args.dataset

    in_channels = DATA_INFO[dataset]["channels"]
    image_res = DATA_INFO[dataset]["resolution"][0]

    seed = args.seed
    seed_all(seed)

    configs_path = os.path.join(args.config_dir, args.dataset + ".json")
    with open(configs_path, "r") as f:
        configs = json.load(f)

    # train parameters
    gettr = partial(get_param, configs_1=configs.get("train", {}), configs_2=args)
    batch_size = gettr("batch_size")
    beta1, beta2 = gettr("beta1"), gettr("beta2")
    lr = gettr("lr")
    epochs = gettr("epochs")
    grad_norm = gettr("grad_norm")
    warmup = gettr("warmup")
    train_device = gettr("train_device")
    eval_device = gettr("eval_device")
    trainloader = get_dataloader(
        dataset, batch_size=batch_size, split="train", val_size=0., random_seed=seed, root=root, pin_memory=True)

    # diffusion parameters
    getdif = partial(get_param, configs_1=configs.get("diffusion", {}), configs_2=args)
    beta_schedule = getdif("beta_schedule")
    beta_start, beta_end = getdif("beta_start"), getdif("beta_end")
    timesteps = getdif("timesteps")
    betas = get_beta_schedule(beta_schedule, beta_start=beta_start, beta_end=beta_end, num_diffusion_timesteps=timesteps)
    model_mean_type = getdif("model_mean_type")
    model_var_type = getdif("model_var_type")
    loss_type = getdif("loss_type")

    diffusion = GaussianDiffusion(
        betas=betas, model_mean_type=model_mean_type, model_var_type=model_var_type, loss_type=loss_type)

    # denoise parameters
    out_channels = 2 * in_channels if model_var_type == "learned" else in_channels

    model = MODEL_LIST[args.model](out_channels=out_channels, **configs["denoise"])
    optimizer = Adam(model.parameters(), lr=lr, betas=(beta1, beta2))
    # Note1: lr_lambda is used to calculate the **multiplicative factor**
    # Note2: index starts at 0
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda t: min((t + 1) / warmup, 1.0))

    hps = {
        "lr": lr,
        "batch_size": batch_size,
        "configs": configs
    }
    hps_info = dict2str(hps)

    chkpt_dir = args.chkpt_dir
    if not os.path.exists(chkpt_dir):
        os.makedirs(chkpt_dir)
    chkpt_path = os.path.join(
        chkpt_dir,
        f"{dataset}_diffusion.pt"
    )
    image_dir = os.path.join(args.image_dir, f"{dataset}")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        diffusion=diffusion,
        epochs=epochs,
        trainloader=trainloader,
        scheduler=scheduler,
        grad_norm=grad_norm,
        device=train_device,
    )
    evaluator = Evaluator(device=eval_device) if args.eval else None
    if args.restart:
        try:
            trainer.restart_from_chkpt(chkpt_path)
        except FileNotFoundError:
            print("Checkpoint file does not exist!")
            print("Starting from scratch...")
    trainer.train(evaluator, chkpt_path=chkpt_path, image_dir=image_dir)
