import torch
from abc import ABC, abstractmethod
from xvfm.optimaltransport import OTSampler
from typing import Union


class FlowModel(ABC):

    @abstractmethod
    def interpolation(self, x_t, t, y=None):
        pass

    @abstractmethod
    def compute_conditional_flow(x0, x1, t, xt):
        pass

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

    def integration(self, model, x_0, steps, device=None, y=None):
        xt = x_0.to(device) if device else x_0
        delta_t = 1.0 / steps

        trajectory = torch.zeros((steps + 1, *xt.shape), device=device)
        trajectory[0] = xt
        
        time_steps = torch.linspace(0, 1, steps, device=device).unsqueeze(1)

        for k in range(steps):
            t = self.pad_t_like_x(time_steps[k].expand(xt.shape[0]), xt)
            v_t = self.interpolation(model, xt, t, y)
            xt = xt + v_t * delta_t
            trajectory[k + 1] = xt

        return trajectory

    @staticmethod
    def sample_noise_like(x):
        return torch.randn_like(x)
    
    @staticmethod
    def pad_t_like_x(t, x):
        if isinstance(t, (float, int)):
            return t
        return t.reshape(-1, *([1] * (x.dim() - 1)))


class CFM(FlowModel):
    def __init__(self, sigma: Union[float, int] = 0.0):
        super().__init__()
        self.sigma = sigma

    @staticmethod
    def interpolation(model, x_t, t, y=None):
        with torch.no_grad():
            if y is None:
                return model(x_t, t)
            else:
                return model(x_t, t, y)

    def compute_mu_t(self, x0, x1, t):
        t = self.pad_t_like_x(t, x0)
        return t * x1 + (1 - t) * x0

    def compute_sigma_t(self, t):
        del t
        return self.sigma

    def sample_xt(self, x0, x1, t, epsilon):
        mu_t = self.compute_mu_t(x0, x1, t)
        sigma_t = self.compute_sigma_t(t)
        sigma_t = self.pad_t_like_x(sigma_t, x0)
        return mu_t + sigma_t * epsilon
        
    @staticmethod
    def compute_conditional_flow(x0, x1, t, xt):
        del t, xt
        return x1 - x0


class OT_CFM(CFM):
    def __init__(self, sigma: Union[float, int] = 0.0):
        super().__init__(sigma)
        self.ot_sampler = OTSampler()

    def sample_location_and_conditional_flow(self, x0, x1, t=None, return_noise=False):
        x0, x1 = self.ot_sampler.sample_plan(x0, x1)
        return super().sample_location_and_conditional_flow(x0, x1, t, return_noise)


class VFM(FlowModel):
    def __init__(self, sigma: Union[float, int] = 0.0, learn_sigma=False):
        super().__init__()
        self.sigma = sigma
        self.learn_sigma = learn_sigma

    def interpolation(self, model, x_t, t, y=None):
        with torch.no_grad():
            if y is None:
                if self.learn_sigma:
                    mu, _ = model(x_t, t)
                else:
                    mu = model(x_t, t)
                v_t = (mu - x_t) / (1 - t)
            else:
                if self.learn_sigma:
                    mu, _ = model(x_t, t, y)
                else:
                    mu = model(x_t, t, y)

                v_t = (mu - x_t) / (1 - t)
            return v_t

    def compute_mu_t(self, x0, x1, t):
        t = self.pad_t_like_x(t, x0)
        return t * x1 + (1 - t) * x0

    def compute_sigma_t(self, t):
        del t
        return self.sigma

    def sample_xt(self, x0, x1, t, epsilon):
        mu_t = self.compute_mu_t(x0, x1, t)
        sigma_t = self.compute_sigma_t(t)
        sigma_t = self.pad_t_like_x(sigma_t, x0)
        return mu_t + sigma_t * epsilon
    
    def compute_conditional_flow(self, x0, x1, t, xt):
        del x0
        t =  self.pad_t_like_x(t, xt)
        return (x1 - xt) / (1 - t)