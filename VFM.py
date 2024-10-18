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


def main():
    savedir = os.path.join(os.getcwd(), "Results/VFM")
    Path(savedir).mkdir(parents=True, exist_ok=True)

    sigma = 0.1
    dim = 2
    batch_size = 256
    patience = 500
    counter = 0
    best_loss = 1e10

    model = MLP(dim=dim, time_varying=True)
    optimizer = torch.optim.Adam(model.parameters())
    FM = CFM(sigma=sigma)
    criterion = torch.nn.MSELoss()

    start = time.time()
    for k in range(50000):
        optimizer.zero_grad()

        x0 = sample_8gaussians(batch_size)
        x1 = sample_moons(batch_size)

        t, xt, _ = FM.sample_location_and_conditional_flow(x0, x1)

        mu_theta = model(torch.cat([xt, t[:, None]], dim=-1))
        loss = criterion(x1, mu_theta)

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_FD = evaluate(xt, x1)
            best_k = k+1
            counter = 0
        else:
            counter += 1
            if counter > patience:
                # print(f"Early stopping at iteration {k}")
                # end = time.time()
                # print(f"{k+1}: Loss [{best_loss:0.3f}]. FD: [{best_FD:.3f}]. Time [{(end - start):0.2f}].")
                # start = end
                
                with torch.no_grad():
                    traj = trajectories(model, sample_8gaussians(1024), steps=100)
                    plot_trajectories(traj=traj, output=f"{savedir}/VFM_{k+1}.png")

                break
            
        loss.backward()
        optimizer.step()

        if (k + 1) % 1000 == 0:
            # end = time.time()
            # print(f"{k+1}: loss {loss.item():0.3f} time {(end - start):0.2f}")
            # start = end
            
            with torch.no_grad():
                traj = trajectories(model, sample_8gaussians(1024), steps=100)
                plot_trajectories(traj=traj, output=f"{savedir}/VFM_{k+1}.png")
                
    if best_k == 0: best_k = 5000
    # print(f"{best_k}: Loss [{best_loss:0.3f}]. FD: [{best_FD:.3f}]. Time [{(end - start):0.2f}].")
    torch.save(model, f"{savedir}/VFM.pt")
    return best_loss, best_FD, best_k
    

if __name__ == "__main__":
    main()
