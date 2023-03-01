import math
import torch
from tqdm import trange
from .fid_score import InceptionStatistics, get_precomputed, calc_fd
from .precision_recall import ManifoldBuilder, Manifold, calc_pr

__all__ = [
    "InceptionStatistics",
    "get_precomputed",
    "calc_fd",
    "ManifoldBuilder",
    "Manifold",
    "calc_pr",
    "Evaluator"
]


class Evaluator:
    def __init__(
            self,
            dataset,
            diffusion=None,
            eval_batch_size=256,
            eval_total_size=50000,
            device=torch.device("cpu")
    ):
        self.diffusion = diffusion
        # inception stats
        self.istats = InceptionStatistics(device=device)
        self.eval_batch_size = eval_batch_size
        self.eval_total_size = eval_total_size
        self.device = device
        self.target_mean, self.target_var = get_precomputed(dataset)

    def eval(self, sample_fn, is_leader=True):
        if is_leader:
            self.istats.reset()
        fid = None
        num_batches = math.ceil(self.eval_total_size / self.eval_batch_size)
        with trange(num_batches, desc="Evaluating FID", disable=not is_leader) as t:
            for i in t:
                if i == len(t) - 1:
                    batch_size = self.eval_total_size % self.eval_batch_size
                else:
                    batch_size = self.eval_batch_size
                x = sample_fn(sample_size=batch_size, diffusion=self.diffusion)
                if is_leader:
                    self.istats(x.to(self.device))
                    if i == len(t) - 1:
                        gen_mean, gen_var = self.istats.get_statistics()
                        fid = calc_fd(gen_mean, gen_var, self.target_mean, self.target_var)
                        t.set_postfix({"fid": fid})
        return {"fid": fid}
