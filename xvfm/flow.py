import torch

from abc import ABC, abstractmethod
from xvfm.prior import Prior
from torchdyn.core import NeuralODE
from xvfm.variational import VariationalDist, GaussianMultinomialDist
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

    @abstractmethod
    def velocity_field(self, x_t, t):
        pass

    def sample_t_and_x_t(self, x_1):
        num_samples = x_1.shape[0]
        t = self.interpolator.sample_t(num_samples).to(x_1.device)
        # x_0 = self.prior.sample(num_samples).view(-1, *x_1.shape[1:]).to(x_1.device)
        x_0 = self.prior.sample(num_samples).to(x_1.device)
        if x_0.shape[1] != x_1.shape[1]:
            x = torch.zeros((num_samples, x_0.shape[1]), device=x_1.device)
            x[:, :self.prior.num_feat] = x_0[:, :self.prior.num_feat]
            idx = self.prior.num_feat
            for i, val in enumerate(self.prior.cat_feat):
                x[:, idx:idx+val] = torch.nn.functional.one_hot(x_1[:, self.prior.num_feat + i].to(torch.int64), num_classes=val)

            x_1 = x

        x_t = self.interpolator.sample_x_t(x_0, x_1, t).to(x_1.device)
        return t, x_t

    def generate(self, num_samples=100, steps=100, device=None, method='euler'):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        xt = self.prior.sample(num_samples).to(device)
        
        if xt.shape[1] == 28:
            xt = xt.view(-1, 1, 28, 28).to(device)
        else:
            xt = xt.to(device)

        if method == 'euler':
            with torch.no_grad():
                delta_t = 1.0 / steps
                trajectory = torch.zeros((steps - 1, *xt.shape), device=device)
                trajectory[0] = xt
                time_steps = torch.linspace(0, 1, steps, device=device).unsqueeze(1)

                for i in range(steps - 2):
                    t = time_steps[i].expand(xt.shape[0], 1)
                    v_t = self.velocity_field(xt, t)
                    xt += v_t * delta_t
                    trajectory[i + 1] = xt

                return trajectory
            
        elif method == 'adaptive':    
            v_t = Velocity(self.variational_dist, self.interpolator)
            node = NeuralODE(v_t, solver="euler", sensitivity="adjoint", atol=1e-4, rtol=1e-4).to(device)
            t = torch.linspace(0, 1, steps, device=device)
            with torch.no_grad():
                return node.trajectory(xt, t_span=t)
            
        else:
            raise ValueError("Invalid method argument")


class VFM(FlowModel):
    def __init__(
            self,
            prior: Prior, 
            variational_dist: VariationalDist, 
            interpolator: Interpolator,
            learn_sigma=False
            ):
        super().__init__(prior, variational_dist, interpolator)
        self.learn_sigma = learn_sigma

    def velocity_field(self, x_t, t):
        posterior = self.variational_dist(x_t, t)

        if isinstance(posterior, tuple):
            gaussian, cat_list = posterior
            mu = torch.zeros_like(x_t, device=x_t.device)
            mu[:, :self.prior.num_feat] = gaussian.mean
            idx = self.prior.num_feat
            for i, val in enumerate(self.prior.cat_feat):
                mu[:, idx:idx+val] = cat_list[i].probs
                idx += val
        else:
            mu = posterior.mean.view(-1, *x_t.shape[1:]).to(x_t.device)
        return self.interpolator.compute_v_t(mu, x_t, t.view(-1, *([1] * (x_t.dim() - 1))))
    

class Velocity(torch.nn.Module):
    def __init__(self, variational_dist, interpolator):
        super(Velocity, self).__init__()
        self.variational_dist = variational_dist
        self.interpolator = interpolator

    def forward(self, t, x_t, args=None):
        mu = self.variational_dist(x_t, t).mean.view(-1, *x_t.shape[1:]).to(x_t.device)
        return self.interpolator.compute_v_t(mu, x_t, t.view(-1, *([1] * (x_t.dim() - 1))))
