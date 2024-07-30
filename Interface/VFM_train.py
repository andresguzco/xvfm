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
    model = MLP(dim=dim, time_varying=True)
    optimizer = torch.optim.Adam(model.parameters())
    FM = CFM(sigma=sigma)
    criterion = torch.nn.MSELoss()

    start = time.time()
    for k in tqdm(range(20000)):
        optimizer.zero_grad()

        x0 = sample_8gaussians(batch_size)
        x1 = sample_moons(batch_size)

        t, xt, _ = FM.sample_location_and_conditional_flow(x0, x1)

        vt = model(torch.cat([xt, t[:, None]], dim=-1))
        loss = criterion(vt, x1)

        loss.backward()
        optimizer.step()

        if (k + 1) % 5000 == 0:
            end = time.time()
            print(f"{k+1}: loss {loss.item():0.3f} time {(end - start):0.2f}")
            start = end
            
            with torch.no_grad():
                traj = trajectories(model, sample_8gaussians(1024), steps=100)
                plot_trajectories(traj=traj, output=f"{savedir}/VFM_{k+1}.png")
                evaluate(traj[-1], sample_moons(1024))

    torch.save(model, f"{savedir}/VFM.pt")
    

if __name__ == "__main__":
    main()