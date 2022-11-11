from .fid_score import InceptionStatistics, get_precomputed, calc_fd
from .precision_recall import ManifoldBuilder, Manifold, calc_pr
import torch

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
            max_eval_count=10000,
            device=torch.device("cpu")
    ):
        self.diffusion = diffusion
        # inception stats
        self.istats = InceptionStatistics(device=device)
        self.eval_batch_size = eval_batch_size
        self.max_eval_count = max_eval_count
        self.device = device
        self.target_mean, self.target_var = get_precomputed(dataset)

    def eval(self, sample_fn):
        self.istats.reset()
        for _ in range(0, self.max_eval_count + self.eval_batch_size, self.eval_batch_size):
            x = sample_fn(self.eval_batch_size, diffusion=self.diffusion)
            self.istats(x.to(self.device))
        gen_mean, gen_var = self.istats.get_statistics()
        return {"fid": calc_fd(gen_mean, gen_var, self.target_mean, self.target_var)}
