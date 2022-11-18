import math
import torch
from .. import diffusion
from ..functions import normal_kl, continuous_gaussian_loglik, flat_mean


class GaussianDiffusion(diffusion.GaussianDiffusion):

    def q_sample(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)
        coef1 = self._extract(self.sqrt_alphas_bar, t, x_0)
        coef2 = self._extract(self.sqrt_one_minus_alphas_bar, t, x_0)
        return coef1 * x_0 + coef2 * noise

    def p_mean_var(self, denoise_fn, x_t, t, clip_denoised, return_pred):
        B, D = x_t.shape
        out = denoise_fn(x_t, t)

        if self.model_var_type == "learned":
            assert all(out.shape == (B, 2 * D))
            out, model_logvar = out.chunk(2, dim=1)
            model_var = torch.exp(model_logvar)
        elif self.model_var_type in ["fixed-small", "fixed-large"]:
            model_var, model_logvar = self._extract(self.fixed_model_var, t, x_t),\
                                      self._extract(self.fixed_model_logvar, t, x_t)
        else:
            raise NotImplementedError(self.model_var_type)

        # calculate the mean estimate
        _clip = lambda x: x  # (lambda x: x.clamp(-3., 3.)) if clip_denoised else (lambda x: x)
        if self.model_mean_type == "mean":  # noqa
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

    # === log likelihood ===
    # bpd: bits per dimension

    def _loss_term_bpd(self, denoise_fn, x_0, x_t, t, clip_denoised, return_pred):
        # calculate L_t
        # t = 0: negative log likelihood of decoder, -\log p(x_0 | x_1)
        # t > 0: variational lower bound loss term, KL term
        true_mean, _, true_logvar = self.q_posterior_mean_var(x_0=x_0, x_t=x_t, t=t)  # noqa
        model_mean, _, model_logvar, pred_x_0 = self.p_mean_var(
            denoise_fn, x_t=x_t, t=t, clip_denoised=clip_denoised, return_pred=True)
        kl = normal_kl(true_mean, true_logvar, model_mean, model_logvar)
        kl = flat_mean(kl) / math.log(2.)  # natural base to base 2
        decoder_nll = continuous_gaussian_loglik(x_0, model_mean, logvar=model_logvar).neg()
        decoder_nll = flat_mean(decoder_nll) / math.log(2.)
        output = torch.where(t.to(kl.device) > 0, kl, decoder_nll)
        return (output, pred_x_0) if return_pred else output
