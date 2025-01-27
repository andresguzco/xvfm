import torch
from abc import ABC, abstractmethod
from torch.nn.functional import one_hot
from torch.distributions import MultivariateNormal, Categorical, Normal, Bernoulli

class Prior(ABC):
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def sample(self, num_samples):
        raise NotImplementedError

class StandardGaussianPrior(Prior):
    def __init__(self, num_feat):
        super(StandardGaussianPrior, self).__init__()
        self.dist = MultivariateNormal(torch.zeros(num_feat), torch.eye(num_feat))
    
    def sample(num_samples):
        return self.dist.sample((num_samples,))

class GaussianMultinomialPrior(Prior):
    def __init__(self, num_feat, cat_feat, task):
        super(GaussianMultinomialPrior, self).__init__()
        self.num_feat = num_feat
        self.cat_feat = cat_feat 
        self.gaussian_dist = MultivariateNormal(torch.zeros(num_feat), torch.eye(num_feat) * 100)

        if sum(cat_feat) != 0:
            self.multinomial_dists = [Categorical(torch.ones(val)/val) for val in cat_feat]
        else:
            self.multinomial_dists = False
        
        if task == 'regression':
            self.cond_dist = Normal(0, 100)
        else:
            self.cond_dist = Bernoulli(0.5)

    def sample(self, num_samples):
        gaussian_samples = self.gaussian_dist.sample((num_samples,))
        samples = []       
        target_sample = self.cond_dist.sample((num_samples, )).unsqueeze(1)
        
        if self.multinomial_dists:
            for i, dist in enumerate(self.multinomial_dists):
                sample = one_hot(dist.sample((num_samples,)), num_classes=self.cat_feat[i])
                samples.append(sample) 
     
            multinomial_samples = torch.cat(samples, dim=1).view(num_samples, -1)
            out = torch.cat([gaussian_samples, multinomial_samples, target_sample], dim=-1)
        else:
            out = torch.cat([gaussian_samples, target_sample], dim=-1)

        return out
