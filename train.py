import os
import json
import torch
from datetime import datetime
from torch.optim import Adam, lr_scheduler
from ddpm_torch import *
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.elastic.multiprocessing import errors
from functools import partial


@errors.record
def main(args):

    distributed = args.distributed

    def logger(msg, **kwargs):
        if not distributed or dist.get_rank() == 0:
            print(msg, **kwargs)

    root = os.path.expanduser(args.root)
    dataset = args.dataset

    in_channels = DATA_INFO[dataset]["channels"]
    image_res = DATA_INFO[dataset]["resolution"]
    image_shape = (in_channels, ) + image_res

    # set seed for all rngs
    seed = args.seed
    seed_all(seed)

    configs_path = os.path.join(args.config_dir, args.dataset + ".json")
    with open(configs_path, "r") as f:
        configs = json.load(f)

    # train parameters
    gettr = partial(get_param, configs_1=configs.get("train", {}), configs_2=args)
    t_cfgs = Configs(**{k: gettr(k) for k in ("batch_size", "beta1", "beta2", "lr", "epochs", "grad_norm", "warmup")})
    t_cfgs.batch_size //= args.num_accum
    train_device = torch.device(args.train_device)
    eval_device = torch.device(args.eval_device)

    # diffusion parameters
    getdif = partial(get_param, configs_1=configs.get("diffusion", {}), configs_2=args)
    d_cfgs = Configs(**{
        k: getdif(k) for k in (
            "beta_schedule",
            "beta_start",
            "beta_end",
            "timesteps",
            "model_mean_type",
            "model_var_type",
            "loss_type"
        )})

    betas = get_beta_schedule(
        d_cfgs.beta_schedule, beta_start=d_cfgs.beta_start, beta_end=d_cfgs.beta_end, timesteps=d_cfgs.timesteps)

    diffusion = GaussianDiffusion(betas=betas, **d_cfgs)

    # denoise parameters
    out_channels = 2 * in_channels if d_cfgs.model_var_type == "learned" else in_channels
    m_cfgs = configs["denoise"]
    m_cfgs["out_channels"] = out_channels
    _model = UNet(**m_cfgs)

    if distributed:
        # check whether torch.distributed is available
        # CUDA devices are required to run with NCCL backend
        assert dist.is_available() and torch.cuda.is_available()
        dist.init_process_group("nccl")
        rank = dist.get_rank()  # global process id across all node(s)
        local_rank = int(os.environ["LOCAL_RANK"])  # local device id on a single node
        torch.cuda.set_device(local_rank)
        _model.cuda()
        model = DDP(_model, device_ids=[local_rank, ])
        train_device = torch.device(f"cuda:{local_rank}")
    else:
        rank = local_rank = 0  # main process by default
        model = _model.to(train_device)

    optimizer = Adam(model.parameters(), lr=t_cfgs.lr, betas=(t_cfgs.beta1, t_cfgs.beta2))
    # Note1: lr_lambda is used to calculate the **multiplicative factor**
    # Note2: index starts at 0
    scheduler = lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda t: min((t + 1) / t_cfgs.warmup, 1.0)) if t_cfgs.warmup > 0 else None

    split = "all" if dataset == "celeba" else "train"
    num_workers = args.num_workers
    trainloader, sampler = get_dataloader(
        dataset, batch_size=t_cfgs.batch_size, split=split, val_size=0., random_seed=seed,
        root=root, drop_last=True, pin_memory=True, num_workers=num_workers, distributed=distributed
    )  # drop_last to have a static input shape; num_workers > 0 to enable asynchronous data loading

    hps = {
        "dataset": dataset,
        "seed": seed,
        "use_ema": args.use_ema,
        "ema_decay": args.ema_decay,
        "num_accum": args.num_accum,
        "train": t_cfgs,
        "denoise": m_cfgs,
        "diffusion": d_cfgs
    }
    timestamp = datetime.now().strftime("%Y-%m-%dT%H%M%S%f")

    chkpt_dir = args.chkpt_dir
    if not os.path.exists(chkpt_dir):
        os.makedirs(chkpt_dir)

    # keep a record of hyperparameter settings used for this experiment run
    with open(os.path.join(chkpt_dir, f"exp_{timestamp}.info"), "w") as f:
        json.dump(hps, f, indent=2)

    chkpt_path = os.path.join(chkpt_dir, args.chkpt_name or f"ddpm_{dataset}.pt")
    chkpt_intv = args.chkpt_intv
    logger(f"Checkpoint will be saved to {os.path.abspath(chkpt_path)}", end=" ")
    logger(f"every {chkpt_intv} epoch(s)")

    image_dir = os.path.join(args.image_dir, f"{dataset}")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    image_intv = args.image_intv
    num_save_images = args.num_save_images
    logger(f"Generated images (x{num_save_images}) will be saved to {os.path.abspath(image_dir)}", end=" ")
    logger(f"every {image_intv} epoch(s)")

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        diffusion=diffusion,
        epochs=t_cfgs.epochs,
        trainloader=trainloader,
        sampler=sampler,
        scheduler=scheduler,
        num_accum=args.num_accum,
        use_ema=args.use_ema,
        grad_norm=t_cfgs.grad_norm,
        shape=image_shape,
        device=train_device,
        chkpt_intv=chkpt_intv,
        image_intv=image_intv,
        num_save_images=num_save_images,
        ema_decay=args.ema_decay,
        rank=rank,
        distributed=distributed
    )
    evaluator = Evaluator(dataset=dataset, device=eval_device) if args.eval else None
    # in case of elastic launch, resume should always be turned on
    resume = args.resume or distributed
    if resume:
        try:
            map_location = {"cuda:0": f"cuda:{local_rank}"} if distributed else train_device
            _chkpt_path = args.chkpt_path or chkpt_path
            trainer.load_checkpoint(_chkpt_path, map_location=map_location)
        except FileNotFoundError:
            logger("Checkpoint file does not exist!")
            logger("Starting from scratch...")

    # use cudnn benchmarking algorithm to select the best conv algorithm
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = True
        logger(f"cuDNN benchmark: ON")

    logger("Training starts...", flush=True)
    trainer.train(evaluator, chkpt_path=chkpt_path, image_dir=image_dir)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--dataset", choices=["mnist", "cifar10", "celeba"], default="cifar10")
    parser.add_argument("--root", default="~/datasets", type=str, help="root directory of datasets")
    parser.add_argument("--epochs", default=50, type=int, help="total number of training epochs")
    parser.add_argument("--lr", default=0.0002, type=float, help="learning rate")
    parser.add_argument("--beta1", default=0.9, type=float, help="beta_1 in Adam")
    parser.add_argument("--beta2", default=0.999, type=float, help="beta_2 in Adam")
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--num-accum", default=1, type=int, help="number of mini-batches before an update")
    parser.add_argument("--timesteps", default=1000, type=int, help="number of diffusion steps")
    parser.add_argument("--beta-schedule", choices=["quad", "linear", "warmup10", "warmup50", "jsd"], default="linear")
    parser.add_argument("--beta-start", default=0.0001, type=float)
    parser.add_argument("--beta-end", default=0.02, type=float)
    parser.add_argument("--model-mean-type", choices=["mean", "x_0", "eps"], default="eps", type=str)
    parser.add_argument("--model-var-type", choices=["learned", "fixed-small", "fixed-large"], default="fixed-large", type=str)
    parser.add_argument("--loss-type", choices=["kl", "mse"], default="mse", type=str)
    parser.add_argument("--num-workers", default=4, type=int, help="number of workers for data loading")
    parser.add_argument("--train-device", default="cuda:0", type=str)
    parser.add_argument("--eval-device", default="cuda:0", type=str)
    parser.add_argument("--image-dir", default="./images/train", type=str)
    parser.add_argument("--image-intv", default=1, type=int)
    parser.add_argument("--num-save-images", default=64, type=int, help="number of images to generate & save")
    parser.add_argument("--config-dir", default="./configs", type=str)
    parser.add_argument("--chkpt-dir", default="./chkpts", type=str)
    parser.add_argument("--chkpt-name", default="", type=str)
    parser.add_argument("--chkpt-intv", default=5, type=int, help="frequency of saving a checkpoint")
    parser.add_argument("--seed", default=1234, type=int, help="random seed")
    parser.add_argument("--resume", action="store_true", help="to resume training from a checkpoint")
    parser.add_argument("--chkpt-path", default="", type=str, help="checkpoint path used to resume training")
    parser.add_argument("--eval", action="store_true", help="whether to evaluate fid during training")
    parser.add_argument("--use-ema", action="store_true", help="whether to use exponential moving average")
    parser.add_argument("--ema-decay", default=0.9999, type=float, help="decay factor of ema")
    parser.add_argument("--distributed", action="store_true", help="whether to use distributed training")

    main(parser.parse_args())
