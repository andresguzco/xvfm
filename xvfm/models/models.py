import torch


class MLP(torch.nn.Module):
    def __init__(self, dim, out_dim=None, w=64, time_varying=False):
        super().__init__()
        self.time_varying = time_varying
        if out_dim is None:
            out_dim = dim
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim + (1 if time_varying else 0), w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, out_dim),
        )

    def forward(self, x_t, t):
        x = torch.cat([x_t, t[:, None]], dim=-1) if self.time_varying else x_t
        return self.net(x)
    

class MLPS(torch.nn.Module):
    def __init__(self, dim, time_varying):
        super(MLPS, self).__init__()
        self.mu = MLP(dim=dim, time_varying=time_varying)
        self.sigma = torch.nn.Parameter(torch.ones(2))

    def forward(self, x, t):
        mu = self.mu(x, t)
        sigma = torch.exp(self.sigma)
        return mu, sigma


class GradModel(torch.nn.Module):
    def __init__(self, action):
        super().__init__()
        self.action = action

    def forward(self, x):
        x = x.requires_grad_(True)
        grad = torch.autograd.grad(torch.sum(self.action(x)), x, create_graph=True)[0]
        return grad[:, :-1]