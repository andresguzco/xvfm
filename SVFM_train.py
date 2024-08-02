import os
import time
import torch
from tqdm import tqdm
from Engine import *
from pathlib import Path


def trajectories(model, x_0, steps):
    xt = x_0
    delta_t = 1 / steps
    trajectory = [xt.cpu().numpy()]
    for k in range(steps):
        t = k / steps * torch.ones(xt.shape[0], 1)
        x1 = model(torch.cat([xt, t], dim=-1))
        v_t = (x1 - xt) / (1 - t)
        xt = xt + v_t * delta_t
        trajectory.append(xt.cpu().numpy())

    trajectory = np.array(trajectory)
    return torch.tensor(trajectory)


class SigmaMLP(MLP):
    def __init__(self, dim, out_dim=None, w=64, time_varying=False):
        super().__init__(dim, out_dim=out_dim, w=w, time_varying=time_varying)
        self.last_filter = torch.nn.ReLU()

    def forward(self, x):
        pred = self.net(x)
        pred = self.last_filter(pred)
        return pred


class SingleParamModel(torch.nn.Module):
    def __init__(self):
        super(SingleParamModel, self).__init__()
        self.param = torch.nn.Parameter(torch.randn(1))

    def forward(self):
        constrained_param = torch.sigmoid(self.param)
        return constrained_param


def main():
    savedir = os.path.join(os.getcwd(), "Results/SVFM")
    Path(savedir).mkdir(parents=True, exist_ok=True)

    dim = 2
    batch_size = 256
    noise = 0.2

    model = MLP(dim=dim, time_varying=True)
    sigma_model = SigmaMLP(dim=0, out_dim=1, time_varying=True)
    # sigma_model = SingleParamModel()

    optimizer = torch.optim.Adam(
        [param for param in model.parameters()] + [param for param in sigma_model.parameters()]
    )

    FM = SVFM()
    criterion = torch.nn.GaussianNLLLoss()


    start = time.time()
    for k in tqdm(range(50000)):
        optimizer.zero_grad()

        x0 = sample_8gaussians(batch_size)
        x1 = sample_moons(batch_size, noise=noise)
        # sigma = sigma_model()

        # t, xt, _ = FM.sample_location_and_conditional_flow(sigma, x0, x1)
        t, xt, _, sigma = FM.sample_location_and_conditional_flow(sigma_model, x0, x1)

        mu_theta = model(torch.cat([xt, t[:, None]], dim=-1))

        # var = torch.ones(batch_size, dim) * (sigma**2)

        # loss = criterion(mu_theta, x1, var)
        loss = criterion(mu_theta, x1, sigma**2)

        loss.backward()
        optimizer.step()
        
        if (k + 1) % 5000 == 0:
            end = time.time()
            print(f"{k+1}: loss {loss.item():0.3f} time {(end - start):0.2f}")
            start = end

            with torch.no_grad():
                traj = trajectories(model, sample_8gaussians(1024), steps=100)
                plot_trajectories(traj=traj.cpu().numpy(), output=f"{savedir}/SVFM_{k+1}.png")
                evaluate(traj[-1].cpu(), sample_moons(1024))
                
    torch.save(model, f"{savedir}/SVFM.pt")
    torch.save(sigma, f"{savedir}/Sigma.pt")


if __name__ == "__main__":
    main()
