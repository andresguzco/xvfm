import os
import time
import torch
import argparse

from tqdm import tqdm
from pathlib import Path
from xvfm.models.models import MLP, MLPS
from xvfm.models.fm import CFM, OT_CFM, VFM
from xvfm.utils import sample_8gaussians, sample_moons, plot_trajectories


MODEL_MAP = {
    'cfm': {'flow': CFM, 'model': MLP},
    'ot': {'flow': OT_CFM, 'model': MLP},
    'vfm': {'flow': VFM, 'model': MLPS}
}

CRITERION_MAP = {
    'cfm': {'default': torch.nn.MSELoss},
    'ot': {'default': torch.nn.MSELoss},
    'vfm': {
        'MSE': torch.nn.MSELoss,
        'SSM': lambda: None,  # Placeholder for SSM loss
        'Gaussian': torch.nn.GaussianNLLLoss
    }
}


def get_args():
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(description='Two Moon Experiment')
    parser.add_argument('--model_type', default='vfm', type=str, help="Type of model: 'cfm', 'ot', 'vfm'")
    parser.add_argument('--device', default=0, type=int, help="CUDA device number, -1 for CPU")
    parser.add_argument('--num_epochs', default=5000, type=int, help="Number of training epochs")
    parser.add_argument('--batch_size', default=256, type=int, help="Training batch size")
    parser.add_argument('--vfm_loss', default='MSE', type=str, help="Loss function for VFM: 'MSE', 'SSM', or 'Gaussian'")
    parser.add_argument('--integration_steps', default=100, type=int, help="Number of steps for integration in trajectory plotting")
    parser.add_argument('--dim', default=2, type=int, help="Dimensionality of the model input")
    parser.add_argument('--sigma', default=0.1, type=float, help="Sigma parameter for flow model")
    parser.add_argument('--save_model', action='store_true', help="Flag to save the trained model")
    return parser.parse_args()


def main(args):
    """Main function to set up the model, criterion, and train."""
    # Check if model_type is valid
    assert args.model_type in MODEL_MAP, "Model type incorrect or not implemented."
    
    device = torch.device("cuda" if torch.cuda.is_available() and args.device >= 0 else "cpu")
    savedir = os.path.join(os.getcwd(), f"results/{args.model_type}")
    Path(savedir).mkdir(parents=True, exist_ok=True)

    # Initialize models and criteria using dictionary mapping
    flow_model = MODEL_MAP[args.model_type]['flow'](args.sigma)
    model = MODEL_MAP[args.model_type]['model'](args.dim, time_varying=True).to(device)
    criterion = CRITERION_MAP[args.model_type][args.vfm_loss if args.model_type == 'vfm' else 'default']()

    optimizer = torch.optim.Adam(model.parameters())
    train_model(args, model, flow_model, criterion, optimizer, device, savedir)

    if args.save_model:
        torch.save(model.state_dict(), f"{savedir}/{args.model_type}.pt")


def train_model(args, model, flow_model, criterion, optimizer, device, savedir):
    """Train the model based on the given configuration."""
    start = time.time()
    for epoch in tqdm(range(args.num_epochs)):
        optimizer.zero_grad()
        x_0 = sample_8gaussians(args.batch_size)
        x_1 = sample_moons(args.batch_size)
        t, x_t, u_t = flow_model.sample_location_and_conditional_flow(x_0, x_1)

        if args.model_type in ['cfm', 'ot']:
            v_t = model(torch.cat([x_t, t[:, None]], dim=-1))
            loss = criterion(v_t, u_t)
        else:  # VFM model
            mu_theta, sigma_theta = model(torch.cat([x_t, t[:, None]], dim=-1))
            if args.vfm_loss == 'MSE':
                loss = criterion(mu_theta, x_1)
            else:
                loss = criterion(mu_theta, x_1, sigma_theta.repeat(x_1.shape[0], 1))

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 1000 == 0:
            end = time.time()
            print(f"{epoch+1}: loss {loss.item():0.3f} time {(end - start):0.2f}")
            start = end
            traj = flow_model.integration(model, sample_8gaussians(1024), steps=args.integration_steps)
            plot_trajectories(traj=traj, output=f"{savedir}/{args.model_type}_{epoch+1}.png")


if __name__ == "__main__":
    args = get_args()
    main(args)