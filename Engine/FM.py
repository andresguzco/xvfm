import torch
from .OT import OTSampler
from typing import Union


def pad_t_like_x(t, x):
    if isinstance(t, (float, int)):
        return t
    return t.reshape(-1, *([1] * (x.dim() - 1)))


class CFM:
    def __init__(self, sigma: Union[float, int] = 0.0):
        self.sigma = sigma

    def compute_mu_t(self, x0, x1, t):
        t = pad_t_like_x(t, x0)
        return t * x1 + (1 - t) * x0

    def compute_sigma_t(self, t):
        del t
        return self.sigma

    def sample_xt(self, x0, x1, t, epsilon):
        mu_t = self.compute_mu_t(x0, x1, t)
        sigma_t = self.compute_sigma_t(t)
        sigma_t = pad_t_like_x(sigma_t, x0)
        return mu_t + sigma_t * epsilon

    def compute_conditional_flow(self, x0, x1, t, xt):
        del t, xt
        return x1 - x0

    def sample_noise_like(self, x):
        return torch.randn_like(x)

    def sample_location_and_conditional_flow(self, x0, x1, t=None, return_noise=False):
        if t is None:
            t = torch.rand(x0.shape[0]).type_as(x0)
        assert len(t) == x0.shape[0], "t has to have batch size dimension"

        eps = self.sample_noise_like(x0)
        xt = self.sample_xt(x0, x1, t, eps)
        ut = self.compute_conditional_flow(x0, x1, t, xt)
        if return_noise:
            return t, xt, ut, eps
        else:
            return t, xt, ut


class OT_CFM(CFM):
    def __init__(self, sigma: Union[float, int] = 0.0):
        super().__init__(sigma)
        self.ot_sampler = OTSampler(method="exact")

    def sample_location_and_conditional_flow(self, x0, x1, t=None, return_noise=False):
        x0, x1 = self.ot_sampler.sample_plan(x0, x1)
        return super().sample_location_and_conditional_flow(x0, x1, t, return_noise)

    
class VFM(CFM):
    def __init__(self, sigma: Union[float, int] = 0.0):
        super().__init__(sigma=sigma)
    
    def compute_conditional_flow(self, x0, x1, t, xt):
        t = pad_t_like_x(t, x1)
        return (x1 - xt) / (1 - t)

class SVFM(VFM):
    def __init__(self, sigma: Union[float, int] = 0.0):
        super().__init__(sigma=sigma)

    def compute_sigma_t(self, t, sigma_model):
        return sigma_model(t[:, None])

    def sample_xt(self, x0, x1, t, epsilon, sigma_model):
        mu_t = self.compute_mu_t(x0, x1, t)
        sigma_t = self.compute_sigma_t(t, sigma_model)
        sigma_t = pad_t_like_x(sigma_t, x0)
        return mu_t + sigma_t * epsilon, sigma_t

    def sample_location_and_conditional_flow(self, sigma_model, x0, x1, t=None, return_noise=False):
        assert sigma_model is not None, "Sigma model has to be provided for SVFM"

        if t is None:
            t = torch.rand(x0.shape[0]).type_as(x0)
        assert len(t) == x0.shape[0], "t has to have batch size dimension"

        eps = self.sample_noise_like(x0)
        xt, sigma = self.sample_xt(x0, x1, t, eps, sigma_model)
        ut = self.compute_conditional_flow(x0, x1, t, xt)
        if return_noise:
            return t, xt, ut, eps, sigma
        else:
            return t, xt, ut, sigma


class SVFM2(VFM):

    def __init__(self, sigma: Union[float, int] = 0.0):
        super().__init__(sigma=sigma)

    def sample_xt(self, x0, x1, t, epsilon, sigma_t):
        mu_t = self.compute_mu_t(x0, x1, t)
        sigma_t = pad_t_like_x(sigma_t, x0)
        return mu_t + sigma_t * epsilon

    def sample_location_and_conditional_flow(self, sigma, x0, x1, t=None, return_noise=False):
        assert sigma is not None, "Sigma has to be provided for SVFM"

        if t is None:
            t = torch.rand(x0.shape[0]).type_as(x0)
        assert len(t) == x0.shape[0], "t has to have batch size dimension"

        eps = self.sample_noise_like(x0)
        xt = self.sample_xt(x0, x1, t, eps, sigma)
        ut = self.compute_conditional_flow(x0, x1, t, xt)
        if return_noise:
            return t, xt, ut, eps
        else:
            return t, xt, ut
