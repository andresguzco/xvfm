import torch
import math
import numpy as np

class Prior:
    def __init__(self):
        self.dist = None

    def sample(self, num_samples):
        return self.dist.sample((num_samples,))


class StandardGaussianPrior(Prior):
    def __init__(self, num_feat):
        super(StandardGaussianPrior, self).__init__()
        self.dist = torch.distributions.MultivariateNormal(torch.zeros(num_feat), 0.2 * torch.eye(num_feat))

class MultiGaussianPrior(StandardGaussianPrior):
    def __init__(self, num_feat):
        super(MultiGaussianPrior, self).__init__(num_feat)
        self.num_feat = num_feat

    def sample(self, num_samples, scale=3.0):
        centers = [
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
            (1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
            (-1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
            (-1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
        ]

        centers = torch.tensor(centers) * scale
        noise = self.dist.sample((num_samples,))
        multi = torch.multinomial(torch.ones(8), num_samples, replacement=True)
        
        data = []
        for i in range(num_samples):
            data.append(centers[multi[i]] + noise[i])

        data = torch.stack(data)

        return data.float()
    


