import math
import torch

class VariationalDist(torch.nn.Module):
    def __init__(self):
        super(VariationalDist, self).__init__()


class GaussianVariationalDist(VariationalDist):
    def __init__(self, posterior_mu_model, posterior_logsigma_model=None):
        super(GaussianVariationalDist, self).__init__()
        self.posterior_mu_model = posterior_mu_model
        if posterior_logsigma_model is not None:
            self.posterior_logsigma_model = posterior_logsigma_model

    def forward(self, x_t, t):
        if t.shape == ():
            t = t.expand(x_t.size(0))
        
        if x_t.dim() > 2:
            mu = self.posterior_mu_model(x_t, t)
            mu = mu.view(-1, math.prod(mu.shape[1:]))
        else:
            mu = self.posterior_mu_model(torch.cat([x_t, t], dim=-1))
        
        identity = torch.eye(mu.size(1)).to(mu.device).unsqueeze(0).expand(mu.size(0), -1, -1)
        if hasattr(self, "posterior_logsigma_model"):
            sigma = torch.exp(self.posterior_logsigma_model(t))
            # if sigma.dim() > 2:
            #     sigma = sigma.view(-1, math.prod(sigma.shape[1:]))
            sigma = sigma.unsqueeze(-1) * identity
        else:
            t = t.unsqueeze(1).expand(-1, mu.size(1), mu.size(1))
            t = torch.clamp(t, 0, 1)
            sigma = (1 - (1 - 0.01) * t) * identity
            # covariance = ((1 - (1 - 0.01) * t)**2) * identity
            # covariance = (1/2) * identity
            # covariance = identity

        return torch.distributions.MultivariateNormal(mu, sigma)
    
    def get_parameters(self):
        if hasattr(self, "posterior_sigma_model"):
            return list(self.posterior_mu_model.parameters()) + list(self.posterior_logsigma_model.parameters())
        else:
            return list(self.posterior_mu_model.parameters())
