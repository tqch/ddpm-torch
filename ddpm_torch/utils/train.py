import os
import re
import math
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa
from torch.utils.data.distributed import DistributedSampler
from torchvision.utils import save_image as _save_image
import weakref
from tqdm import tqdm
from functools import partial
from contextlib import nullcontext


class DummyScheduler:
    @staticmethod
    def step():
        pass

    def load_state_dict(self, state_dict):
        pass

    @staticmethod
    def state_dict():
        return None


class RunningStatistics:
    def __init__(self, **kwargs):
        self.count = 0
        self.stats = []
        for k, v in kwargs.items():
            self.stats.append((k, v or 0))
        self.stats = dict(self.stats)

    def reset(self):
        self.count = 0
        for k in self.stats:
            self.stats[k] = 0

    def update(self, n, **kwargs):
        self.count += n
        for k, v in kwargs.items():
            self.stats[k] = self.stats.get(k, 0) + v

    def extract(self):
        avg_stats = []
        for k, v in self.stats.items():
            avg_stats.append((k, v / self.count))
        return dict(avg_stats)

    def __repr__(self):
        out_str = "Count(s): {}\n"
        out_str += "Statistics:\n"
        for k in self.stats:
            out_str += f"\t{k} = {{{k}}}\n"  # double curly-bracket to escape
        return out_str.format(self.count, **self.stats)


save_image = partial(_save_image, normalize=True, value_range=(-1., 1.))


class Trainer:
    def __init__(
            self,
            model,
            optimizer,
            diffusion,
            epochs,
            trainloader,
            sampler=None,
            scheduler=None,
            num_accum=1,
            use_ema=False,
            grad_norm=1.0,
            shape=None,
            device=torch.device("cpu"),
            chkpt_intv=5,
            image_intv=1,
            num_samples=64,
            ema_decay=0.9999,
            distributed=False,
            rank=0,  # process id for distributed training
            dry_run=False
    ):
        self.model = model
        self.optimizer = optimizer
        self.diffusion = diffusion
        self.epochs = epochs
        self.start_epoch = 0
        self.trainloader = trainloader
        self.sampler = sampler
        if shape is None:
            shape = next(iter(trainloader))[0].shape[1:]
        self.shape = shape
        self.scheduler = DummyScheduler() if scheduler is None else scheduler

        self.num_accum = num_accum
        self.grad_norm = grad_norm
        self.device = device
        self.chkpt_intv = chkpt_intv
        self.image_intv = image_intv
        self.num_samples = num_samples

        if distributed:
            assert sampler is not None
        self.distributed = distributed
        self.rank = rank
        self.dry_run = dry_run
        self.is_leader = rank == 0
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))

        # maintain a process-specific generator
        self.generator = torch.Generator(device).manual_seed(8191 + self.rank)

        self.sample_seed = 131071 + self.rank  # process-specific seed

        self.use_ema = use_ema
        if use_ema:
            if isinstance(model, DDP):
                self.ema = EMA(model.module, decay=ema_decay)
            else:
                self.ema = EMA(model, decay=ema_decay)
        else:
            self.ema = nullcontext()

        self.stats = RunningStatistics(loss=None)

    @property
    def timesteps(self):
        return self.diffusion.timesteps

    def get_input(self, x):
        x = x.to(self.device)
        return {
            "x_0": x,
            "t": torch.empty((x.shape[0],), dtype=torch.int64, device=self.device).random_(
                to=self.timesteps, generator=self.generator),
            "noise": torch.empty_like(x).normal_(generator=self.generator)
        }

    def loss(self, x):
        loss = self.diffusion.train_losses(self.model, **self.get_input(x))
        assert loss.shape == (x.shape[0],)
        return loss

    def step(self, x, global_steps=1):
        # Note: For DDP models, the gradients collected from different devices are averaged rather than summed.
        # See https://pytorch.org/docs/1.12/generated/torch.nn.parallel.DistributedDataParallel.html
        # Mean-reduced loss should be used to avoid inconsistent learning rate issue when number of devices changes.
        loss = self.loss(x).mean()
        loss.div(self.num_accum).backward()  # average over accumulated mini-batches
        if global_steps % self.num_accum == 0:
            # gradient clipping by global norm
            # Note: In the official TF1.15+TPU implementation (clip_by_global_norm + CrossShardOptimizer)
            # the gradient clipping operation is performed at shard level (i.e., TPU core or device level)
            # see also https://github.com/tensorflow/tensorflow/blob/v1.15.0/tensorflow/python/tpu/tpu_optimizer.py#L114-L118
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            # adjust learning rate every step (e.g. warming up)
            self.scheduler.step()
            if self.use_ema and hasattr(self.ema, "update"):
                self.ema.update()
        loss = loss.detach()
        if self.distributed:
            dist.reduce(loss, dst=0, op=dist.ReduceOp.SUM)  # synchronize losses
            loss.div_(self.world_size)
        self.stats.update(x.shape[0], loss=loss.item() * x.shape[0])

    def sample_fn(self, sample_size=None, noise=None, diffusion=None, sample_seed=None):
        if noise is None:
            shape = (sample_size // self.world_size,) + self.shape
        else:
            shape = noise.shape
        if diffusion is None:
            diffusion = self.diffusion
        with self.ema:
            sample = diffusion.p_sample(
                denoise_fn=self.model, shape=shape,
                device=self.device, noise=noise, seed=sample_seed)
        if self.distributed:
            # equalizes GPU memory usages across all processes within the same process group
            sample_list = [torch.zeros(shape, device=self.device) for _ in range(self.world_size)]
            dist.all_gather(sample_list, sample)
            sample = torch.cat(sample_list, dim=0)
        assert sample.grad is None
        return sample

    def train(self, evaluator=None, chkpt_path=None, image_dir=None):
        nrow = math.floor(math.sqrt(self.num_samples))
        if self.num_samples:
            assert self.num_samples % self.world_size == 0, "Number of samples should be divisible by WORLD_SIZE!"

        if self.dry_run:
            self.start_epoch, self.epochs = 0, 1

        global_steps = 0
        for e in range(self.start_epoch, self.epochs):
            self.stats.reset()
            self.model.train()
            results = dict()
            if isinstance(self.sampler, DistributedSampler):
                self.sampler.set_epoch(e)
            with tqdm(self.trainloader, desc=f"{e + 1}/{self.epochs} epochs", disable=not self.is_leader) as t:
                for i, x in enumerate(t):
                    if isinstance(x, (list, tuple)):
                        x = x[0]  # unconditional model -> discard labels
                    global_steps += 1
                    self.step(x.to(self.device), global_steps=global_steps)
                    t.set_postfix(self.current_stats)
                    results.update(self.current_stats)
                    if self.dry_run and not global_steps % self.num_accum:
                        break

            if not (e + 1) % self.image_intv and self.num_samples and image_dir:
                self.model.eval()
                x = self.sample_fn(sample_size=self.num_samples, sample_seed=self.sample_seed).cpu()
                if self.is_leader:
                    save_image(x, os.path.join(image_dir, f"{e + 1}.jpg"), nrow=nrow)

            if not (e + 1) % self.chkpt_intv and chkpt_path:
                self.model.eval()
                if evaluator is not None:
                    eval_results = evaluator.eval(self.sample_fn, is_leader=self.is_leader)
                else:
                    eval_results = dict()
                results.update(eval_results)
                if self.is_leader:
                    self.save_checkpoint(chkpt_path, epoch=e + 1, **results)

            if self.distributed:
                dist.barrier()  # synchronize all processes here

    @property
    def trainees(self):
        roster = ["model", "optimizer"]
        if self.use_ema:
            roster.append("ema")
        if self.scheduler is not None:
            roster.append("scheduler")
        return roster

    @property
    def current_stats(self):
        return self.stats.extract()

    def load_checkpoint(self, chkpt_path, map_location):
        chkpt = torch.load(chkpt_path, map_location=map_location)
        for trainee in self.trainees:
            try:
                getattr(self, trainee).load_state_dict(chkpt[trainee])
            except RuntimeError:
                _chkpt = chkpt[trainee]["shadow"] if trainee == "ema" else chkpt[trainee]
                for k in list(_chkpt.keys()):
                    if k.startswith("module."):
                        _chkpt[k.split(".", maxsplit=1)[1]] = _chkpt.pop(k)
                getattr(self, trainee).load_state_dict(chkpt[trainee])
            except AttributeError:
                continue
        self.start_epoch = chkpt["epoch"]

    def save_checkpoint(self, chkpt_path, **extra_info):
        chkpt = []
        for k, v in self.named_state_dicts():
            chkpt.append((k, v))
        for k, v in extra_info.items():
            chkpt.append((k, v))
        if "epoch" in extra_info:
            chkpt_path = re.sub(r"(_\d+)?\.pt", f"_{extra_info['epoch']}.pt", chkpt_path)
        torch.save(dict(chkpt), chkpt_path)

    def named_state_dicts(self):
        for k in self.trainees:
            yield k, getattr(self, k).state_dict()


class EMA:
    """
    exponential moving average
    inspired by:
    [1] https://github.com/fadel/pytorch_ema
    [2] https://github.com/tensorflow/tensorflow/blob/v2.9.1/tensorflow/python/training/moving_averages.py#L281-L685
    """

    def __init__(self, model, decay=0.9999):
        shadow = []
        refs = []
        for k, v in model.named_parameters():
            if v.requires_grad:
                shadow.append((k, v.detach().clone()))
                refs.append((k, weakref.ref(v)))
        self.shadow = dict(shadow)
        self._refs = dict(refs)
        self.decay = decay
        self.num_updates = -1
        self.backup = None

    def update(self):
        self.num_updates += 1
        decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        for k, _ref in self._refs.items():
            assert _ref() is not None, "referenced object no longer exists!"
            self.shadow[k] += (1 - decay) * (_ref().data - self.shadow[k])

    def apply(self):
        self.backup = dict([
            (k, _ref().detach().clone()) for k, _ref in self._refs.items()])
        for k, _ref in self._refs.items():
            _ref().data.copy_(self.shadow[k])

    def restore(self):
        for k, _ref in self._refs.items():
            _ref().data.copy_(self.backup[k])
        self.backup = None

    def __enter__(self):
        self.apply()

    def __exit__(self, *exc):
        self.restore()

    def state_dict(self):
        return {
            "decay": self.decay,
            "shadow": self.shadow,
            "num_updates": self.num_updates
        }

    @property
    def extra_states(self):
        return {"decay", "num_updates"}

    def load_state_dict(self, state_dict, strict=True):
        _dict_keys = set(self.__dict__["shadow"]).union(self.extra_states)
        dict_keys = set(state_dict["shadow"]).union(self.extra_states)
        incompatible_keys = set.symmetric_difference(_dict_keys, dict_keys) \
            if strict else set.difference(_dict_keys, dict_keys)
        if incompatible_keys:
            raise RuntimeError(
                "Key mismatch!\n"
                f"Missing key(s): {', '.join(set.difference(_dict_keys, dict_keys))}."
                f"Unexpected key(s): {', '.join(set.difference(dict_keys, _dict_keys))}"
            )
        self.__dict__.update(state_dict)


class ModelWrapper(nn.Module):
    def __init__(
            self,
            model,
            pre_transform=None,
            post_transform=None
    ):
        super().__init__()
        self._model = model
        self.pre_transform = pre_transform
        self.post_transform = post_transform

    def forward(self, x, *args, **kwargs):
        if self.pre_transform is not None:
            x = self.pre_transform(x)
        out = self._model(x, *args, **kwargs)
        if self.post_transform is not None:
            out = self.post_transform(out)
        return out
