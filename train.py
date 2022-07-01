import os
import json
import torch
from torch.optim import Adam, lr_scheduler
import matplotlib as mpl
from ddpm_torch import *


mpl.rcParams["figure.dpi"] = 144


MODEL_LIST = {
    "unet": UNet
}


if __name__ == "__main__":
    from argparse import ArgumentParser
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
    parser.add_arugment("--num-save-images", default=64, type=int)
    parser.add_argument("--config-dir", default="./configs", type=str)
    parser.add_argument("--chkpt-dir", default="./chkpts", type=str)
    parser.add_argument("--chkpt-intv", default=5, type=int)
    parser.add_argument("--log-dir", default="./logs", type=str)
    parser.add_argument("--seed", default=1234, type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--eval", action="store_true")

    args = parser.parse_args()

    root = os.path.expanduser(args.root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = args.dataset

    in_channels = DATA_INFO[dataset]["channels"]
    image_res = DATA_INFO[dataset]["resolution"][0]

    # set seed for all rngs
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
    chkpt_intv = args.chkpt_intv
    print(f"Checkpoint will be saved to {os.path.abspath(chkpt_path)}", end="")
    print(f"every {chkpt_intv} epochs")

    image_dir = os.path.join(args.image_dir, f"{dataset}")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    num_save_images = args.num_save_images
    print(f"Generated images (x{num_save_images}) will be saved to {os.path.abspath(image_dir)}")

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        diffusion=diffusion,
        epochs=epochs,
        trainloader=trainloader,
        scheduler=scheduler,
        grad_norm=grad_norm,
        device=train_device,
        chkpt_intv=chkpt_intv,
        num_save_images=num_save_images
    )
    evaluator = Evaluator(dataset=dataset, device=eval_device) if args.eval else None
    if args.resume:
        try:
            trainer.resume_from_chkpt(chkpt_path)
        except FileNotFoundError:
            print("Checkpoint file does not exist!")
            print("Starting from scratch...")

    # use cudnn benchmarking algorithm to select the best conv algorithm
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = True
        print(f"cuDNN benchmark: ON")

    print("Training starts...", flush=True)
    trainer.train(evaluator, chkpt_path=chkpt_path, image_dir=image_dir)
