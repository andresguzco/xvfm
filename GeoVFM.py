import os
import time
import torch
from tqdm import tqdm
from Engine import *
from pathlib import Path
from torch.autograd import Function


class CustomLoss(Function):
    @staticmethod
    def forward(ctx, x, mu, var):
        ctx.save_for_backward(x, mu, var)
        loss = 0.5 * ((x - mu) ** 2 / var + torch.log(var))
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        x, mu, var = ctx.saved_tensors
        grad_mu = mu - x
        grad_sigma2 = (0.5 * ((x - mu) ** 2 - var)) / (var ** 2)
        fisher_mu = 1 / var
        fisher_sigma2 = 1 / (2 * var ** 2)
        natural_grad_mu = grad_mu / fisher_mu
        natural_grad_sigma2 = grad_sigma2 / fisher_sigma2
        return None, natural_grad_mu, natural_grad_sigma2


def Trajectories(model, x_0, steps):
    xt = x_0
    delta_t = 1 / steps
    trajectory = [xt.cpu().numpy()]
    for k in range(steps):
        t = k / steps * torch.ones(xt.shape[0], 1)
        x1, _ = model(torch.cat([xt, t], dim=-1))
        v_t = (x1 - xt) / (1 - t)
        xt = xt + v_t * delta_t
        trajectory.append(xt.cpu().numpy())
    trajectory = np.array(trajectory)
    return torch.tensor(trajectory)


class MLPS(torch.nn.Module):
    def __init__(self, dim, time_varying):
        super(MLPS, self).__init__()
        self.mu = MLP(dim=dim, time_varying=time_varying)
        self.sigma = torch.nn.Parameter(torch.ones(2))
        self.pos_filter = torch.nn.ReLU()

    def forward(self, x, params=None):
        mu = self.mu(x)
        sigma = self.pos_filter(self.sigma)
        return mu, sigma

def main():
    savedir = os.path.join(os.getcwd(), "Results/GeoVFM")
    Path(savedir).mkdir(parents=True, exist_ok=True)

    dim = 2
    batch_size = 256
    noise = 0.2
    patience = 500
    counter = 0
    best_loss = 1e10
    best_FD = 1e10

    model = MLPS(dim=dim, time_varying=True)

    optimizer = torch.optim.Adam(model.parameters())

    FM = CFM(sigma=noise)
    criterion = torch.nn.GaussianNLLLoss()

    # start = time.time()

    for k in range(50000):
        optimizer.zero_grad()

        x0 = sample_8gaussians(batch_size)
        x1 = sample_moons(batch_size, noise=noise)

        t, xt, _ = FM.sample_location_and_conditional_flow(x0, x1)

        mu_theta, sigma_theta = model(torch.cat([xt, t[:, None]], dim=-1))

        loss = criterion(mu_theta, x1, sigma_theta.repeat(x1.shape[0], 1))
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_FD = evaluate(xt, x1)
            best_k = k+1
        else:
            counter += 1
            if counter > patience:
                # end = time.time()
                # print(f"{k+1}: Loss [{best_loss:0.3f}]. FD: [{best_FD:.3f}]. Time [{(end - start):0.2f}].")
                # start = end
                
                with torch.no_grad():
                    traj = Trajectories(model, sample_8gaussians(1024), steps=100)
                    plot_trajectories(traj=traj, output=f"{savedir}/GeoFM_{k+1}.png")
                    evaluate(traj[-1], sample_moons(1024))

                break

        loss.backward()
        optimizer.step()

        if (k + 1) % 1000 == 0:
            # end = time.time()
            # print(f"{k+1}: loss {loss.item():0.6f} time {(end - start):0.2f}")
            # start = end

            with torch.no_grad():
                traj = Trajectories(model, sample_8gaussians(1024), steps=100)
                plot_trajectories(traj=traj.cpu().numpy(), output=f"{savedir}/{k+1}.png")
                # evaluate(traj[-1].cpu(), sample_moons(1024))

    if best_k == 0: best_k = 5000
    # print(f"{best_k}: Loss [{best_loss:0.3}]. FD: [{best_FD:.3f}]. Time [{(end - start):0.2f}].")
    torch.save(model, f"{savedir}/GeoVFM.pt")
    return best_loss, best_FD, best_k


if __name__ == "__main__":
    main()