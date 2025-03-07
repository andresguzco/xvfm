import torch
from torch.distributions import Normal, MultivariateNormal, Categorical

class GuassianMultinomial(torch.nn.Module):
    def __init__(self, num_feat, classes, task):
        super(GuassianMultinomial, self).__init__()
        self.num_feat = num_feat
        self.classes = classes
        self.task_type = task

    def forward(self, res, x, t):
        llk = lambda dist, x: -1. * dist.log_prob(x).mean()

        mu = res[:, :self.num_feat]
        identity = torch.eye(mu.size(1)).to(mu.device).unsqueeze(0).expand(mu.size(0), -1, -1)
        scale = (1 - (1 - 0.01)* t.unsqueeze(1)**2) 
        sigma = scale * identity
        normal = MultivariateNormal(mu, sigma)
        a = llk(normal, x[:, :self.num_feat])

        b = 0
        if sum(self.classes) > 0:
            idx = self.num_feat
            for i, val in enumerate(self.classes):
                logits_pred = res[:, idx:idx+val]
                slice = x[:, self.num_feat + i]
                
                cat = Categorical(logits_pred)
                b += llk(cat, slice.to(torch.int64))
                idx += val

        slice_target = x[:, -1].view(-1, 1).float()
        if self.task_type == 'regression':
            logits_target = res[:, -1].view(-1, 1)
            target = Normal(logits_target, scale)
        else:
            logits_target = res[:, -2:]
            target = Categorical(logits_target)
        
        c = llk(target, slice_target)
        return a + b + c