import os
import time
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from Engine import *
from pathlib import Path
from torchdyn.core import NeuralODE
from torchdyn.datasets import generate_moons

def main():
    savedir = os.path.join(os.getcwd(), "Results/VFM")
    Path(savedir).mkdir(parents=True, exist_ok=True)

    sigma = 0.1
    dim = 2
    batch_size = 256

    model = MLP(dim=dim, time_varying=True)

    optimizer = torch.optim.Adam(model.parameters())
    FM = CFM(sigma=sigma)
    criterion = torch.nn.GaussianNLLLoss()


    start = time.time()
    for k in tqdm(range(20000)):
        optimizer.zero_grad()

        x0 = sample_8gaussians(batch_size)
        x1 = sample_moons(batch_size)

        t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)

        vt = model(torch.cat([xt, t[:, None]], dim=-1))

        var = torch.full_like(x1, sigma**2)
        var.requires_grad = False

        loss = criterion(vt, ut, var)

        loss.backward()
        optimizer.step()

        if (k + 1) % 5000 == 0:
            end = time.time()
            print(f"{k+1}: loss {loss.item():0.3f} time {(end - start):0.2f}")
            start = end
            node = NeuralODE(torch_wrapper(model), solver="euler")
            with torch.no_grad():
                traj = node.trajectory(
                    sample_8gaussians(1024),
                    t_span=torch.linspace(0, 1, 100),
                )
                plot_trajectories(traj=traj.cpu().numpy(), output=f"{savedir}/SVFM_{k+1}.png")
            
            evaluate(traj[-1].cpu(), sample_moons(1024))
                
    torch.save(model, f"{savedir}/SVFM_mu.pt")


if __name__ == "__main__":
    main()