import os
import time
import math
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from Engine import *
from pathlib import Path
from torchdyn.core import NeuralODE
from torchdyn.datasets import generate_moons

def main():
    savedir = os.path.join(os.getcwd(), "Results/SG")
    Path(savedir).mkdir(parents=True, exist_ok=True)

    sigma = 0.1
    dim = 2
    batch_size = 256
    g_t = math.sqrt(1.0)
    var = torch.ones(batch_size, dim, requires_grad=False) * sigma**2

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

        # Calculate s_t^theta(x) = grad_x v_t^theta(x)
        xt.requires_grad_(True)
        vt = model(torch.cat([xt, t[:, None]], dim=-1))
        st = torch.autograd.grad(outputs=vt, inputs=xt, grad_outputs=torch.ones_like(vt), create_graph=True)[0]

        # Compute u_tilde
        v_tilde = vt + (g_t ** 2 / 2) * st
        u_tilde = ut + (g_t ** 2 / 2) * st

        # Loss computation
        loss = criterion(v_tilde, u_tilde, var)

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
                plot_trajectories(traj=traj.cpu().numpy(), output=f"{savedir}/SG_{k+1}.png")
            
            evaluate(traj[-1].cpu(), sample_moons(1024))
                
    torch.save(model, f"{savedir}/SG.pt")


if __name__ == "__main__":
    main()