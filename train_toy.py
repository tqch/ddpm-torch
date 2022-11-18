import os
import torch
import numpy as np
from torch.optim import Adam, lr_scheduler
from ddpm_torch.utils import seed_all, infer_range
from ddpm_torch.toy import *


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", choices=["gaussian8", "gaussian25", "swissroll"], default="gaussian8")
    parser.add_argument("--size", default=100000, type=int)
    parser.add_argument("--root", default="~/datasets", type=str, help="root directory of datasets")
    parser.add_argument("--epochs", default=100, type=int, help="total number of training epochs")
    parser.add_argument("--lr", default=0.001, type=float, help="learning rate")
    parser.add_argument("--beta1", default=0.9, type=float, help="beta_1 in Adam")
    parser.add_argument("--beta2", default=0.999, type=float, help="beta_2 in Adam")
    parser.add_argument("--lr-warmup", default=0, type=int, help="number of warming-up epochs")
    parser.add_argument("--batch-size", default=1000, type=int)
    parser.add_argument("--timesteps", default=100, type=int, help="number of diffusion steps")
    parser.add_argument("--beta-schedule", choices=["quad", "linear", "warmup10", "warmup50", "jsd"], default="linear")
    parser.add_argument("--beta-start", default=0.001, type=float)
    parser.add_argument("--beta-end", default=0.2, type=float)
    parser.add_argument("--model-mean-type", choices=["mean", "x_0", "eps"], default="eps", type=str)
    parser.add_argument("--model-var-type", choices=["learned", "fixed-small", "fixed-large"], default="fixed-large", type=str)  # noqa
    parser.add_argument("--loss-type", choices=["kl", "mse"], default="mse", type=str)
    parser.add_argument("--image-dir", default="./images/train", type=str)
    parser.add_argument("--chkpt-dir", default="./chkpts", type=str)
    parser.add_argument("--chkpt-intv", default=5, type=int, help="frequency of saving a checkpoint")
    parser.add_argument("--eval-intv", default=1, type=int)
    parser.add_argument("--seed", default=1234, type=int, help="random seed")
    parser.add_argument("--resume", action="store_true", help="to resume training from a checkpoint")
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--mid-features", default=128, type=int)
    parser.add_argument("--num-temporal-layers", default=3, type=int)

    args = parser.parse_args()

    # set seed
    seed_all(args.seed)

    # prepare toy data
    in_features = 2
    dataset = args.dataset
    data_size = args.size
    root = os.path.expanduser(args.root)
    batch_size = args.batch_size
    num_batches = data_size // batch_size
    trainloader = DataStreamer(dataset, batch_size=batch_size, num_batches=num_batches)

    # training parameters
    device = torch.device(args.device)
    epochs = args.epochs

    # diffusion parameters
    beta_schedule = args.beta_schedule
    beta_start, beta_end = args.beta_start, args.beta_end
    timesteps = args.timesteps
    betas = get_beta_schedule(
        beta_schedule, beta_start=beta_start, beta_end=beta_end, timesteps=timesteps)
    model_mean_type = args.model_mean_type
    model_var_type = args.model_var_type
    loss_type = args.loss_type
    diffusion = GaussianDiffusion(
        betas=betas, model_mean_type=model_mean_type, model_var_type=model_var_type, loss_type=loss_type)

    # model parameters
    out_features = 2 * in_features if model_var_type == "learned" else in_features
    mid_features = args.mid_features
    model = Decoder(in_features, mid_features, args.num_temporal_layers)
    model.to(device)

    # training parameters
    lr = args.lr
    beta1, beta2 = args.beta1, args.beta2
    optimizer = Adam(model.parameters(), lr=lr, betas=(beta1, beta2))

    # checkpoint path
    chkpt_dir = args.chkpt_dir
    if not os.path.exists(chkpt_dir):
        os.makedirs(chkpt_dir)
    chkpt_path = os.path.join(chkpt_dir, f"ddpm_{dataset}.pt")

    # set up image directory
    image_dir = os.path.join(args.image_dir, f"{dataset}")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    # scheduler
    warmup = args.lr_warmup
    scheduler = lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda t: min((t + 1) / warmup, 1.0)) if warmup > 0 else None

    # load trainer
    grad_norm = 0  # gradient global clipping is disabled
    eval_intv = args.eval_intv
    chkpt_intv = args.chkpt_intv
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        diffusion=diffusion,
        epochs=epochs,
        trainloader=trainloader,
        scheduler=scheduler,
        grad_norm=grad_norm,
        device=device,
        eval_intv=eval_intv,
        chkpt_intv=chkpt_intv
    )

    # load evaluator
    max_eval_count = min(data_size, 30000)
    eval_batch_size = min(max_eval_count, 30000)
    xlim, ylim = infer_range(trainloader.dataset)
    value_range = (xlim, ylim)
    true_data = iter(trainloader)
    evaluator = Evaluator(
        true_data=np.concatenate([
            next(true_data) for _ in range(max_eval_count//eval_batch_size)
        ]), eval_batch_size=eval_batch_size, max_eval_count=max_eval_count, value_range=value_range)

    if args.resume:
        try:
            trainer.load_checkpoint(chkpt_path)
        except FileNotFoundError:
            print("Checkpoint file does not exist!")
            print("Starting from scratch...")

    trainer.train(evaluator, chkpt_path=chkpt_path, image_dir=image_dir, xlim=xlim, ylim=ylim)
