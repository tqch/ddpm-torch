import numpy as np
import torch
from ..functions import normal_kl, discretized_gaussian_loglik


def flat_mean(x, start_dim=1):
    reduce_dim = [i for i in range(start_dim, x.ndim)]
    return torch.mean(x, dim=reduce_dim)


def flat_sum(x, start_dim=1):
    reduce_dim = [i for i in range(start_dim, x.ndim)]
    return torch.sum(x, dim=reduce_dim)


def _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, warmup_frac):
    betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    warmup_time = int(num_diffusion_timesteps * warmup_frac)
    betas[:warmup_time] = np.linspace(beta_start, beta_end, warmup_time, dtype=np.float64)
    return betas


def get_beta_schedule(beta_schedule, beta_start, beta_end, num_diffusion_timesteps):
    if beta_schedule == 'quad':
        betas = np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64) ** 2
    elif beta_schedule == 'linear':
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == 'warmup10':
        betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.1)
    elif beta_schedule == 'warmup50':
        betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.5)
    elif beta_schedule == 'const':
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class GaussianDiffusion:
    def __init__(
            self,
            betas,
            model_mean_type,
            model_var_type,
            loss_type,
            cutoff
    ):
        assert isinstance(betas, np.ndarray) and betas.dtype == np.float64
        assert (betas > 0).all() and (betas <= 1).all()
        self.betas = betas
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.cutoff = cutoff

        self.timesteps = len(betas)

        alphas = 1 - betas
        self.alphas_bar = np.cumprod(alphas)
        self.alphas_bar_prev = np.concatenate([np.ones(1, dtype=np.float64), self.alphas_bar[:-1]])

        # q(x_t | x_{t-1})
        self.sqrt_alphas = np.sqrt(alphas)

        # q(x_t | x_0)
        self.sqrt_alphas_bar = np.sqrt(self.alphas_bar)

        # q(x_{t-1} | x_t, x_0)
        # refer to the formula 1-3 in README.md
        self.sqrt_alphas_bar_prev = np.sqrt(self.alphas_bar_prev)
        self.sqrt_one_minus_alphas_bar = np.sqrt(1 - self.alphas_bar)
        self.sqrt_recip_alphas_bar = np.sqrt(1. / self.alphas_bar)
        self.sqrt_recip_m1_alphas_bar = np.sqrt(1. / self.alphas_bar - 1.)  # m1: minus 1
        self.posterior_var = betas * (1 - self.alphas_bar_prev) / (1 - self.alphas_bar)
        self.posterior_logvar_clipped = np.log(np.concatenate([
            np.array([self.posterior_var[1], ], dtype=np.float64), self.posterior_var[1:]]))
        self.posterior_mean_coef1 = betas * self.sqrt_alphas_bar_prev / (1 - self.alphas_bar)
        self.posterior_mean_coef2 = np.sqrt(alphas) * (1 - self.alphas_bar_prev) / (1 - self.alphas_bar)

    @staticmethod
    def _extract(arr, t, ndim):
        B = len(t)
        out = torch.tensor(arr, dtype=torch.float32)[t]
        return out.reshape((B,) + (1,) * (ndim - 1))

    def q_mean_var(self, x_0, t):
        ndim = x_0.ndim
        mean = self._extract(self.sqrt_alphas_bar, t, ndim=ndim) * x_0
        var = self._extract(1. - self.alphas_bar, t, ndim=ndim)
        logvar = self._extract(self.sqrt_one_minus_alphas_bar, t, ndim=ndim)
        return mean, var, logvar

    def q_sample(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)
        ndim = x_0.ndim
        coef1 = self._extract(self.sqrt_alphas_bar, t, ndim=ndim).to(x_0.device)
        coef2 = self._extract(self.sqrt_one_minus_alphas_bar, t, ndim=ndim).to(x_0.device)
        return coef1 * x_0 + coef2 * noise

    def q_posterior_mean_var(self, x_0, x_t, t):
        ndim = x_0.ndim
        posterior_mean_coef1 = self._extract(self.posterior_mean_coef1, t, ndim=ndim).to(x_0.device)
        posterior_mean_coef2 = self._extract(self.posterior_mean_coef2, t, ndim=ndim).to(x_0.device)
        posterior_mean = posterior_mean_coef1 * x_0 + posterior_mean_coef2 * x_t
        posterior_var = self._extract(self.posterior_var, t, ndim=ndim)
        posterior_logvar = self._extract(self.posterior_logvar_clipped, t, ndim=ndim).to(x_0.device)
        return posterior_mean, posterior_var, posterior_logvar

    def p_mean_var(self, denoise_fn, x_t, t, clip_denoised, return_pred):
        B, D = x_t.shape
        ndim = x_t.ndim
        out = denoise_fn(x_t, t)

        if self.model_var_type == "learned":
            assert all(out.shape == (B, 2 * D))
            out, model_logvar = out.chunk(2, dim=1)
            model_var = torch.exp(model_logvar)
        elif self.model_var_type in ["fixed-small", "fixed-large"]:
            model_var, model_logvar = {
                "fixed-large": (self.betas, np.log(np.concatenate([np.array([self.posterior_var[1]]), self.betas[1:]]))),
                "fixed-small": (self.posterior_var, self.posterior_logvar_clipped)
            }[self.model_var_type]
            model_var, model_logvar = self._extract(model_var, t, ndim=ndim), self._extract(model_logvar, t, ndim=ndim)
            model_var, model_logvar = model_var.to(x_t.device), model_logvar.to(x_t.device)
        else:
            raise NotImplementedError(self.model_var_type)

        # calculate the mean estimate
        _clip = lambda x: x  # lambda x: x.clamp(-3., 3.) if clip_denoised else lambda x: x
        if self.model_mean_type == "mean":
            pred_x_0 = _clip(self._pred_x_0_from_mean(x_t=x_t, mean=out, t=t))
            model_mean = out
        elif self.model_mean_type == "x_0":
            pred_x_0 = _clip(out)
            model_mean, *_ = self.q_posterior_mean_var(x_0=pred_x_0, x_t=x_t, t=t)
        elif self.model_mean_type == "eps":
            pred_x_0 = _clip(self._pred_x_0_from_eps(x_t=x_t, eps=out, t=t))
            model_mean, *_ = self.q_posterior_mean_var(x_0=pred_x_0, x_t=x_t, t=t)
        else:
            raise NotImplementedError(self.model_mean_type)

        if return_pred:
            return model_mean, model_var, model_logvar, pred_x_0
        else:
            return model_mean, model_var, model_logvar

    def _pred_x_0_from_mean(self, x_t, mean, t):
        ndim = x_t.ndim
        coef1 = self._extract(self.posterior_mean_coef1, t, ndim=ndim).to(x_t.device)
        coef2 = self._extract(self.posterior_mean_coef2, t, ndim=ndim).to(x_t.device)
        return mean / coef1 - coef2 / coef1 * x_t

    def _pred_x_0_from_eps(self, x_t, eps, t):
        ndim = x_t.ndim
        coef1 = self._extract(self.sqrt_recip_alphas_bar, t, ndim=ndim).to(x_t.device)
        coef2 = self._extract(self.sqrt_recip_m1_alphas_bar, t, ndim=ndim).to(x_t.device)
        return coef1 * x_t - coef2 * eps

    # === sample ===

    def p_sample_step(self, denoise_fn, x_t, t, clip_denoised=True, return_pred=False):
        ndim = x_t.ndim
        model_mean, _, model_logvar, pred_x_0 = self.p_mean_var(
            denoise_fn, x_t, t, clip_denoised=clip_denoised, return_pred=True)
        noise = torch.randn_like(x_t)
        nonzero_mask = (t > 0).reshape((-1,) + (1,) * (ndim - 1)).to(x_t)
        sample = model_mean + nonzero_mask * torch.exp(0.5 * model_logvar) * noise
        return (sample, pred_x_0) if return_pred else sample

    def p_sample(self, denoise_fn, shape, device=torch.device("cpu"), noise=None):
        B, *_ = shape
        t = torch.ones(B, dtype=torch.int64)
        t.fill_(self.timesteps - 1)
        if noise is None:
            x_t = torch.randn(shape, device=device)
        else:
            x_t = noise.to(device)
        for _ in range(self.timesteps - 1, -1, -1):
            x_t = self.p_sample_step(denoise_fn, x_t, t)
            t -= 1
        return x_t

    def p_sample_progressive(self, denoise_fn, shape, device=torch.device("cpu"), noise=None, pred_freq=50):
        B, *_ = shape
        t = torch.ones(B, dtype=torch.int64)
        t.fill_(self.timesteps - 1)
        if noise is None:
            x_t = torch.randn(shape, device=device)
        else:
            x_t = noise.to(device)
        L = self.timesteps // pred_freq
        preds = torch.zeros((B, L,) + shape[1:], dtype=torch.float32)
        idx = L
        with torch.no_grad():
            for i in range(self.timesteps - 1, -1, -1):
                x_t, pred = self.p_sample_step(denoise_fn, x_t, t, return_pred=True)
                t -= 1
                if (i + 1) % pred_freq == 0:
                    idx -= 1
                    preds[:, idx] = pred.cpu()
        return x_t.cpu(), preds

    # === log likelihood ===
    # bpd: bits per dimension

    def _loss_term_bpd(self, denoise_fn, x_0, x_t, t, clip_denoised, return_pred):
        # calculate L_t
        # t = 0: negative log likelihood of decoder, -\log p(x_0 | x_1)
        # t > 0: variational lower bound loss term, KL term
        true_mean, _, true_logvar = self.q_posterior_mean_var(x_0=x_0, x_t=x_t, t=t)
        model_mean, _, model_logvar, pred_x_0 = self.p_mean_var(
            denoise_fn, x_t=x_t, t=t, clip_denoised=clip_denoised, return_pred=True)
        kl = normal_kl(true_mean, true_logvar, model_mean, model_logvar)
        kl = flat_mean(kl) / np.log(2.)  # natural base to base 2
        decoder_nll = -discretized_gaussian_loglik(
            x_0, model_mean, log_scale=0.5 * model_logvar, cutoff=self.cutoff)
        decoder_nll = flat_mean(decoder_nll) / np.log(2.)
        output = torch.where(t.to(kl.device) > 0, kl, decoder_nll)
        return (output, pred_x_0) if return_pred else output

    def train_losses(self, denoise_fn, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, noise=noise)

        # calculate the loss
        # kl: weighted
        # mse: unweighted
        if self.loss_type == "kl":
            losses = self._loss_term_bpd(
                denoise_fn, x_0=x_0, x_t=x_t, t=t, clip_denoised=False, return_pred=False)
        elif self.loss_type == "mse":
            assert self.model_var_type != "learned"
            if self.model_mean_type == "mean":
                target = self.q_posterior_mean_var(x_0=x_0, x_t=x_t, t=t)[0]
            elif self.model_mean_type == "x_0":
                target = x_0
            elif self.model_mean_type == "eps":
                target = noise
            else:
                raise NotImplementedError(self.model_mean_type)
            model_out = denoise_fn(x_t, t)
            losses = flat_mean((target - model_out).pow(2))
        else:
            raise NotImplementedError(self.loss_type)

        return losses

    def _prior_bpd(self, x_0):
        B, T = len(x_0), self.timesteps
        T_mean, _, T_logvar = self.q_mean_var(x_0=x_0, t=(T - 1) * torch.ones([B, ], dtype=torch.int64))
        kl_prior = normal_kl(T_mean, T_logvar, mean2=0., logvar2=0.)
        return flat_mean(kl_prior) / np.log(2.)

    def calc_all_bpd(self, denoise_fn, x_0, clip_denoised=True):
        B, T = x_0.shape, self.timesteps
        t = torch.ones([B, ], dtype=torch.int64)
        t.fill_(T - 1)
        losses = torch.zeros([B, T], dtype=torch.float32)
        mses = torch.zeros([B, T], dtype=torch.float32)

        for i in range(T - 1, -1, -1):
            x_t = self.q_sample(x_0, t=t)
            loss, pred_x_0 = self._loss_term_bpd(
                denoise_fn, x_0, x_t=x_t, t=t, clip_denoised=clip_denoised, return_pred=True)
            losses[:, i] = loss
            mses[:, i] = flat_mean((pred_x_0 - x_0).pow(2))

        prior_bpd = self._prior_bpd(x_0)
        total_bpd = torch.sum(losses, dim=1) + prior_bpd
        return total_bpd, losses, prior_bpd, mses
