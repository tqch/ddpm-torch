import os
import torch
import numpy as np
from toy_model import Decoder
from torch.optim import Adam, lr_scheduler
from toy_data import DataStreamer
from utils import seed_all
from train_utils import Trainer, Evaluator, DummyScheduler, infer_range
from diffusion import GaussianDiffusion, get_beta_schedule


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", choices=["gaussian8", "gaussian25", "swissroll"])
    parser.add_argument("--size", default=100000, type=int)
    parser.add_argument("--root", default="~/datasets", type=str)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--lr-warmup", default=0, type=int)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.999, type=float)
    parser.add_argument("--batch-size", default=1000, type=int)
    parser.add_argument("--timesteps", default=1000, type=int)
    parser.add_argument("--beta-schedule", choices=["quad", "linear", "warmup10", "warmup50", "jsd"], default="linear")
    parser.add_argument("--beta-start", default=0.0001, type=float)
    parser.add_argument("--beta-end", default=0.02, type=float)
    parser.add_argument("--model-mean-type", choices=["mean", "x_0", "eps"], default="eps", type=str)
    parser.add_argument("--model-var-type", choices=["learned", "fixed-small", "fixed-large"], default="fixed-large", type=str)
    parser.add_argument("--loss-type", choices=["kl", "mse"], default="mse", type=str)
    parser.add_argument("--fig-dir", default="./figs", type=str)
    parser.add_argument("--chkpt-dir", default="./chkpts", type=str)
    parser.add_argument("--seed", default=1234, type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--mid-features", default=128, type=int)
    parser.add_argument("--num-temporal-layers", default=1, type=int)

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
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    epochs = args.epochs

    # diffusion parameters
    beta_schedule = args.beta_schedule
    beta_start, beta_end = args.beta_start, args.beta_end
    timesteps = args.timesteps
    betas = get_beta_schedule(
        beta_schedule, beta_start=beta_start, beta_end=beta_end, num_diffusion_timesteps=timesteps)
    model_mean_type = args.model_mean_type
    model_var_type = args.model_var_type
    loss_type = args.loss_type
    diffusion = GaussianDiffusion(
        betas=betas, model_mean_type=model_mean_type, model_var_type=model_var_type, loss_type=loss_type)

    # model parameters
    out_features = 2 * in_features if model_var_type == "learned" else in_features
    mid_features = args.mid_features
    model = Decoder(in_features, mid_features, args.num_temporal_layers)

    # training parameters
    lr = args.lr
    beta1, beta2 = args.beta1, args.beta2
    optimizer = Adam(model.parameters(), lr=lr, betas=(beta1, beta2))

    # checkpoint
    chkpt_dir = args.chkpt_dir
    if not os.path.exists(chkpt_dir):
        os.makedirs(chkpt_dir)
    chkpt_path = os.path.join(chkpt_dir, f"{dataset}_diffusion.pt")

    # figure
    fig_dir = os.path.join(args.fig_dir, f"{dataset}")
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    # scheduler
    warmup_epochs = args.lr_warmup
    if warmup_epochs:
        lr_lambda = lambda t: min((t + 1) / warmup_epochs, 1.0)
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    else:
        scheduler = DummyScheduler()

    # load trainer
    grad_norm = 0  # gradient global clipping is disabled
    eval_intv = 1
    chkpt_intv = 10
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
    xlim, ylim = infer_range(trainloader, precision=1)
    value_range = (xlim, ylim)
    true_data = iter(trainloader)
    evaluator = Evaluator(
        true_data=np.concatenate([
            next(true_data) for _ in range(max_eval_count//eval_batch_size)
        ]), eval_batch_size=eval_batch_size, max_eval_count=max_eval_count, value_range=value_range)

    if args.resume:
        try:
            trainer.resume_from_chkpt(chkpt_path)
        except FileNotFoundError:
            print("Checkpoint file does not exist!")
            print("Starting from scratch...")

    trainer.train(evaluator, chkpt_path=chkpt_path, image_dir=fig_dir, xlim=xlim, ylim=ylim)
