import os
import torch
import torch.nn as nn
from utils import RunningStatistics, save_image
from metrics.fid_score import InceptionStatistics, get_precomputed, fid
from tqdm import tqdm


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
                    save_image(x, os.path.join(image_dir, f"{e+1}.jpg"))
            # adjust learning rate every epoch before checkpoint
            self.scheduler.step()
            if (e+1) % self.chkpt_intv and chkpt_path:
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
            dataset,
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