import os
import time
import torch
import argparse
from pathlib import Path
from xvfm.models.models import MLP, MLPS
from xvfm.models.fm import CFM, OT_CFM, VFM
from xvfm.utils import (
    sample_8gaussians, 
    sample_moons, 
    plot_trajectories
)


def main(args):
    assert args.model_type in ['cfm', 'ot', 'vfm'], "Model type incorrect or not implemented."

    savedir = os.path.join(os.getcwd(), f"results/{args.model_type}")
    Path(savedir).mkdir(parents=True, exist_ok=True)

    if args.model_type == 'cfm':
        flow_model = CFM(args.sigma)
        model = MLP(args.dim, time_varying=True)
        criterion = torch.nn.MSELoss()

    elif args.model_type == 'ot':
        flow_model = OT_CFM(args.sigma)
        model = MLP(args.dim, time_varying=True)
        criterion = torch.nn.MSELoss()

    else : # if args.model_type == 'vfm'
        flow_model = VFM(args.sigma)
        model = MLPS(args.dim, time_varying=True)

        if args.vfm_loss == 'MSE':
            criterion = torch.nn.MSELoss()
        elif args.vfm_loss == 'SSM': # Sufficient Statistics Matching 
            ... 
        else: # Gaussian Loss
            criterion = torch.nn.GaussianNLLLoss()

    optimizer = torch.optim.Adam(model.parameters())

    start = time.time()
    for k in range(5000):
        optimizer.zero_grad()

        x_0 = sample_8gaussians(args.batch_size)
        x_1 = sample_moons(args.batch_size)

        t, x_t, u_t = flow_model.sample_location_and_conditional_flow(x_0, x_1)

        if args.model_type in ['cfm', 'ot']:
            v_t = model(torch.cat([x_t, t[:, None]], dim=-1))
            loss = criterion(v_t, u_t)
        else:
            mu_theta, sigma_theta = model(torch.cat([x_t, t[:, None]], dim=-1))
            if args.vfm_loss == 'MSE':
                loss = criterion(mu_theta, x_1)
            else:
                loss = criterion(mu_theta, x_1, sigma_theta.repeat(x_1.shape[0], 1))

        loss.backward()
        optimizer.step()

        if (k + 1) % 1000 == 0:
            end = time.time()
            print(f"{k+1}: loss {loss.item():0.3f} time {(end - start):0.2f}")

            start = end
            traj = flow_model.integration(model, sample_8gaussians(1024), steps=args.integration_steps)
            plot_trajectories(traj=traj, output=f"{savedir}/{args.model_type}_{k+1}.png")
                
    if args.save_model:
        torch.save(model, f"{savedir}/{args.model_type}.pt")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Two Moon Experiment')

    parser.add_argument('--model_type', default='vfm', type=str)
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--num_epochs', default=5000, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--vfm_loss', default='MSE', type=str)
    parser.add_argument('--integration_steps', default=100, type=int)
    # parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--dim', default=2, type=int)
    parser.add_argument('--sigma', default=0.1, type=float)
    parser.add_argument('--save_model', default=False, type=bool)

    args = parser.parse_args()
    main(args)
