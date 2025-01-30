import torch


class Interpolator:
    def __init__(self):
        pass

    @staticmethod
    def sample_t(num_samples):
        return torch.rand((num_samples, 1))

class OTInterpolator(Interpolator):
    def __init__(self, sigma_min=0.1):
        super().__init__()
        self.sigma_min = sigma_min

    def sample_x_t(self, x_0, x_1, t):
        x_t = t * x_1 + (1 - t) * x_0
        x_t += torch.randn_like(x_t) * self.sigma_min
        return x_t

    def compute_v_t(self, mu, x, t):
        return (mu - (1 - self.sigma_min) * x) # / (1 - (1 - self.sigma_min) * t)
