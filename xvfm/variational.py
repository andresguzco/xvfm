import math
import torch

class VariationalDist(torch.nn.Module):
    def __init__(self):
        super(VariationalDist, self).__init__()


class GaussianVariationalDist(VariationalDist):
    def __init__(self, posterior_mu_model):
        super(GaussianVariationalDist, self).__init__()
        self.posterior_mu_model = posterior_mu_model

    def forward(self, x_t, t):
        if t.shape == ():
            t = t.expand(x_t.size(0))

        mu = self.posterior_mu_model(x_t, t)
        if mu.dim() > 2:
            mu = mu.view(-1, math.prod(mu.shape[1:]))
            
        identity = torch.eye(mu.size(1)).to(mu.device).unsqueeze(0).expand(mu.size(0), -1, -1)
        t = t.unsqueeze(1).expand(-1, mu.size(1), mu.size(1))
        t = torch.clamp(t, 0, 1)
        covariance = ((1 - (1 - 0.01) * t)**2) * identity
        # covariance = (1/2) * identity
        # covariance = identity
        return torch.distributions.MultivariateNormal(mu, covariance)