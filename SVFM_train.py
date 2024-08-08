import os
import time
import torch
from tqdm import tqdm
from Engine import *
from pathlib import Path


class MLPD(torch.nn.Module):
    def __init__(self, dim, time_varying):
        super(MLPD, self).__init__()
        self.mu = MLP(dim=dim, time_varying=time_varying)
        self.sigma = MLP(dim=dim, out_dim=2, time_varying=time_varying)
        self.pos_filter = torch.nn.ReLU()
        

    def forward(self, x):
        mu = self.mu(x)
        sigma = self.sigma(x)
        sigma = self.pos_filter(sigma)
        return mu, sigma



def trajectories(model, x_0, steps):
    xt = x_0
    delta_t = 1 / steps
    trajectory = [xt.cpu().numpy()]
    for k in range(steps):
        t = k / steps * torch.ones(xt.shape[0], 1)
        x1, _ = model(torch.cat([xt, t], dim=-1))
        print()

        v_t = (x1 - xt) / (1 - t)
        xt = xt + v_t * delta_t
        trajectory.append(xt.cpu().numpy())

    trajectory = np.array(trajectory)
    return torch.tensor(trajectory)


class MLPS(torch.nn.Module):
    def __init__(self, dim, time_varying):
        super(MLPS, self).__init__()
        self.mu = MLP(dim=dim, time_varying=time_varying)
        self.sigma = torch.nn.Parameter(torch.rand(2))
        # init_params = torch.rand(3)
        # self.sigma = torch.nn.Parameter(torch.tensor([[init_params[0], init_params[1]*0.5], [init_params[1]*0.5, init_params[2]]]))
        self.pos_filter = torch.nn.ReLU()

    def forward(self, x):
        mu = self.mu(x)
        sigma = self.pos_filter(self.sigma)
        return mu, sigma


def main():
    savedir = os.path.join(os.getcwd(), "Results/SVFM")
    Path(savedir).mkdir(parents=True, exist_ok=True)

    dim = 2
    batch_size = 256
    noise = 0.2

    model = MLPS(dim=dim, time_varying=True)

    optimizer = torch.optim.Adam(model.parameters())

    FM = CFM(sigma=noise)
    criterion = torch.nn.GaussianNLLLoss()
    # criterion = MultivariateNormalNLLLoss()


    start = time.time()
    for k in tqdm(range(20000)):
        optimizer.zero_grad()

        x0 = sample_8gaussians(batch_size)
        x1 = sample_moons(batch_size, noise=noise)

        t, xt, _ = FM.sample_location_and_conditional_flow(x0, x1)

        mu_theta, sigma_theta = model(torch.cat([xt, t[:, None]], dim=-1))

        loss = criterion(mu_theta, x1, sigma_theta.repeat(x1.shape[0], 1))
        # loss = criterion(mu_theta, x1, sigma_theta)
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

            print(f"Avg. X_1 from p: [{torch.mean(x1):.4f}]")
            print(f"Avg. X_1 form q: [{torch.mean(mu_theta):.4f}]")
            print(f"Std. X_1 from p: [{torch.std(x1[:, 0]).item():.4f}, {torch.std(x1[:, 1]).item():.4f}]")
            print(f"Std. X_1 from q: [{sigma_theta[0]:.4f}, {sigma_theta[1]:.4f}]") 
                
    torch.save(model, f"{savedir}/SVFM.pt")


if __name__ == "__main__":
    main()
