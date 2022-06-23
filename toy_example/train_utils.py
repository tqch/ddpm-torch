import os
import math
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from utils import save_scatterplot
from functions import discrete_klv2d, hist2d


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
            avg_stats.append((k, v/self.count))
        return dict(avg_stats)

    def __repr__(self):
        out_str = "Count(s): {}\n"
        out_str += "Statistics:\n"
        for k in self.stats:
            out_str += f"\t{k} = {{{k}}}\n"  # double curly-bracket to escape
        return out_str.format(self.count, **self.stats)


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
            shape=None,
            grad_norm=0,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            eval_intv=1,  # evaluate every {eval_intv} epochs
            chkpt_intv=10  # save a checkpoint every {chkpt_intv} epochs
    ):

        self.model = model
        self.optimizer = optimizer
        self.diffusion = diffusion
        self.epochs = epochs
        self.start_epoch = 0
        self.trainloader = trainloader
        if shape is None:
            shape = next(iter(trainloader)).shape[1:]
        self.shape = tuple(shape)
        self.scheduler = scheduler
        self.grad_norm = grad_norm
        self.device = device
        self.eval_intv = eval_intv
        self.chkpt_intv = chkpt_intv

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
        if self.grad_norm:
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_norm)
        self.optimizer.step()
        self.stats.update(B, loss=loss.item() * B)

    def train(self, evaluator=None, chkpt_path=None, image_dir=None, **plot_kwargs):
        
        def sample_fn(n):
            shape = (n, ) + self.shape
            with torch.no_grad():
                sample = self.diffusion.p_sample(
                    denoise_fn=self.model, shape=shape, device=self.device, noise=None)
            return sample.cpu().numpy()
        
        for e in range(self.start_epoch, self.epochs):
            self.stats.reset()
            self.model.train()
            with tqdm(self.trainloader, desc=f"{e+1}/{self.epochs} epochs") as t:
                for i, x in enumerate(t):
                    self.step(x.to(self.device))
                    t.set_postfix(self.current_stats) 
                    if i == len(self.trainloader) - 1:
                        eval_results = dict()
                        if (e+1) % self.eval_intv == 0:
                            self.model.eval()
                            if evaluator is not None:
                                eval_results = evaluator.eval(sample_fn)
                        x_gen = eval_results.pop("x_gen", None)
                        if x_gen is not None and image_dir:
                            save_scatterplot(
                                os.path.join(image_dir, f"{e+1}.jpg"), x_gen, **plot_kwargs)
                        results = dict()
                        results.update(self.current_stats)
                        results.update(eval_results)
                        t.set_postfix(results)
            # adjust learning rate every epoch before checkpoint
            self.scheduler.step()
            if (e+1) % self.chkpt_intv == 0 and chkpt_path:
                self.save_checkpoint(chkpt_path, epoch=e+1, **results)

    @property
    def current_stats(self):
        return self.stats.extract()

    def resume_from_chkpt(self, chkpt_path):
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
            true_data,
            eval_batch_size=500,
            max_eval_count=30000,
            value_range=(-3, 3),
            eps=1e-9
    ):
        self.eval_batch_size = eval_batch_size
        self.max_eval_count = max_eval_count
        self.bins = math.floor(math.sqrt(self.max_eval_count//10))
        self.value_range = value_range
        self.eps = eps
        self.true_hist = self.get_histogram(true_data)
        self.true_hist.setflags(write=False)  # make true_hist read-only
    
    def get_histogram(self, data):
        hist = 0
        for i in range(0, len(data), self.eval_batch_size):
            hist += hist2d(
                data[i:(i+self.eval_batch_size)], bins=self.bins, value_range=self.value_range)
        hist /= np.sum(hist) + self.eps  # avoid zero-division
        return hist
        
    def eval(self, sample_fn):
        x_gen = []
        gen_hist = 0
        with torch.no_grad():
            for _ in range(0, self.max_eval_count + self.eval_batch_size, self.eval_batch_size):
                with torch.no_grad():
                    x_gen.append(sample_fn(self.eval_batch_size))
                    gen_hist += hist2d(
                        x_gen[-1], bins=self.bins, value_range=self.value_range)
        gen_hist /= np.sum(gen_hist) + self.eps
        return {
            "kld": discrete_klv2d(gen_hist, self.true_hist),
            "x_gen": np.concatenate(x_gen, axis=0)
        }


def infer_range(dataset, precision=2):
    p = precision
    # infer proper x,y axes limits for evaluation/plotting
    xlim = np.array([-np.inf, np.inf])
    ylim = np.array([-np.inf, np.inf])
    _approx_clip = lambda x, y, z: np.clip([
        math.floor(p*x), math.ceil(p*y)], *z)
    for bch in dataset:
        xlim = _approx_clip(bch[:, 0].min(), bch[:, 0].max(), xlim)
        ylim = _approx_clip(bch[:, 1].min(), bch[:, 1].max(), ylim)
    return xlim / p, ylim / p
