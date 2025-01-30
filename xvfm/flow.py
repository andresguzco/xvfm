import torch

from abc import ABC, abstractmethod
from xvfm.prior import Prior
from xvfm.variational import VariationalDist
from xvfm.interpolator import Interpolator


class FlowModel(torch.nn.Module, ABC):
    def __init__(
            self, 
            prior: Prior, 
            variational_dist: VariationalDist, 
            interpolator: Interpolator
            ):
        super().__init__()
        self.prior = prior
        self.variational_dist = variational_dist
        self.interpolator = interpolator
        self.task = variational_dist.task

    @abstractmethod
    def velocity_field(self, x_t, t):
        pass

    def sample_t_and_x_t(self, x_1, y):
        num_samples = x_1.shape[0]
        t = self.interpolator.sample_t(num_samples).view(-1, 1).to(x_1.device)
        x_0 = self.prior.sample(num_samples).to(x_1.device)

        x = torch.zeros((num_samples, x_0.shape[1]), device=x_1.device)
        x[:, :self.prior.num_feat] = x_0[:, :self.prior.num_feat]
        
        if sum(self.prior.cat_feat) != 0:
            idx = self.prior.num_feat
            for i, val in enumerate(self.prior.cat_feat):
                x[:, idx:idx+val] = torch.nn.functional.one_hot(
                    x_1[:, self.prior.num_feat + i].to(torch.int64), num_classes=val
                    )

        x[:, -1] = y.view(-1, 1) 
        x_t = self.interpolator.sample_x_t(x_0, x, t).to(x_1.device)
        return t, x_t

    def generate(self, num_samples=100, steps=100, device=None, y=None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        xt = self.prior.sample(num_samples).to(device)

        with torch.no_grad():
            delta_t = 1.0 / steps
            trajectory = torch.zeros((steps - 1, *xt.shape), device=device)
            time_steps = torch.linspace(0, 1, steps, device=device).unsqueeze(1)

            trajectory[0] = xt
            for i in range(steps - 2):
                t = time_steps[i].expand(xt.shape[0], 1)
                v_t = self.velocity_field(xt, t, y)
                xt += v_t * delta_t
                trajectory[i + 1] = xt

            return trajectory


class VFM(FlowModel):
    def __init__(self, prior: Prior, variational_dist: VariationalDist, interpolator: Interpolator):
        super(VFM, self).__init__(prior, variational_dist, interpolator)

    def velocity_field(self, x_t, t, y=None):
        posterior = self.variational_dist(x_t, t, y)
        gaussian, cat_list, target = posterior
        
        mu = torch.zeros_like(x_t, device=x_t.device)
        mu[:, :self.prior.num_feat] = gaussian.mean
        
        if cat_list != False:
            idx = self.prior.num_feat
            for i, val in enumerate(self.prior.cat_feat):
                mu[:, idx:idx+val] = cat_list[i].probs
                idx += val   

        mu[:, -1] = target.mean if self.task == 'regression' else target.probs

        return self.interpolator.compute_v_t(mu, x_t, t)
    

class AlternativeFlow(FlowModel):
    def __init__(self, prior: Prior, variational_dist: VariationalDist, interpolator: Interpolator):
        super(AlternativeFlow, self).__init__(prior, variational_dist, interpolator)

    def velocity_field(self, x_t, t, y=None):
        mu = self.variational_dist(x_t, t, y)
        return self.interpolator.compute_v_t(mu, x_t, t)
