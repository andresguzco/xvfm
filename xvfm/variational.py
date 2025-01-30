import torch


class VariationalDist(torch.nn.Module):
    def __init__(self):
        super(VariationalDist, self).__init__()


class GaussianVariationalDist(VariationalDist):
    def __init__(self, model):
        super(GaussianVariationalDist, self).__init__()
        self.model = model

    def forward(self, x_t, t):
        if t.shape == ():
            t = t.expand(x_t.size(0))
        
        mu = self.model(torch.cat([x_t, t], dim=-1))
        identity = torch.eye(mu.size(1)).to(mu.device).unsqueeze(0).expand(mu.size(0), -1, -1)
        t = t.unsqueeze(1).expand(-1, mu.size(1), mu.size(1))
        t = torch.clamp(t, 0, 1)
        sigma = ((1 - (1 - 0.01) * t)**2) * identity

        return torch.distributions.MultivariateNormal(mu, sigma)
    
    def get_parameters(self):
        return list(self.posterior_mu_model.parameters())


class GaussianMultinomialDist(VariationalDist):
    def __init__(self, model, num_feat, cat_feat, task):
        super(GaussianMultinomialDist, self).__init__()
        self.model = model 
        self.num_feat = num_feat
        self.cat_feat = cat_feat
        self.task = task

    def forward(self, x_t, t, y=None):
        res = self.model(x_t, t, y)

        mu = res[:, :self.num_feat]

        identity = torch.eye(mu.size(1)).to(mu.device).unsqueeze(0).expand(mu.size(0), -1, -1)
        scale = (1 - (1 - 0.01)* t.unsqueeze(1)**2) 
        sigma = scale * identity
        
        normal_dist = torch.distributions.MultivariateNormal(mu, sigma)
        
        if sum(self.cat_feat) != 0:
            cat_dists, cum_sum = [], self.num_feat
            for val in self.cat_feat:
                slice = res[:, cum_sum:cum_sum + val]
                cat_dists.append(torch.distributions.Categorical(slice))
                cum_sum += val
        else:
            cat_dists = False
        
        if self.task == 'regression':
            target = torch.distributions.Normal(res[:, -1], scale.squeeze())
        else:
            target = torch.distributions.Bernoulli(res[:, -1])

        return normal_dist, cat_dists, target

    def get_parameters(self):
        return list(self.model.parameters())


class Alternative(VariationalDist):
    def __init__(self, model, num_feat, cat_feat, task):
        super(Alternative, self).__init__()
        self.model = model 
        self.num_feat = num_feat
        self.cat_feat = cat_feat
        self.task = task

    def forward(self, x_t, t, y=None):
        return self.model(x_t, t, y)

    def get_parameters(self):
        return list(self.model.parameters())