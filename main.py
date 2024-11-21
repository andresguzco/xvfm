import os
import torch
import argparse

from torchvision import datasets, transforms

from tqdm import tqdm
from pathlib import Path
from xvfm.flow import VFM
from xvfm.unet import UNetModel
from data.utils import evaluate
from xvfm.prior import StandardGaussianPrior, MultiGaussianPrior
from xvfm.models import MLP
from data.two_moons import generate_two_moons
from xvfm.variational import GaussianVariationalDist
from xvfm.interpolator import OTInterpolator


CRITERION_MAP = {
    'MSE': lambda posterior, x_1: torch.mean((x_1 - posterior.mean) ** 2),
    'SSM': lambda: None,  # Placeholder for SSM loss,
    'Gaussian': lambda posterior, x_1: -1 * posterior.log_prob(x_1).mean()
}


def get_args():
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(description='Two Moon Experiment')
    parser.add_argument('--model_type', default='vfm', type=str, help="Type of model: 'cfm', 'vfm'")
    parser.add_argument('--num_epochs', default=10, type=int, help="Number of training epochs")
    parser.add_argument('--batch_size', default=256, type=int, help="Training batch size")
    parser.add_argument('--lr', default=1e-3, type=float, help="Learning rate for optimizer")
    parser.add_argument('--loss_fn', default='MSE', type=str, help="Loss function for VFM: 'MSE', 'SSM', or 'Gaussian'")
    parser.add_argument('--learn_sigma', type=bool, default=False, help="Flag to learn sigma in VFM")
    parser.add_argument('--int_method', default='euler', help="Integration method for trajectory plotting: 'euler', 'adaptive'")
    parser.add_argument('--integration_steps', default=100, type=int, help="Number of steps for integration in trajectory plotting")
    parser.add_argument('--sigma', default=0.1, type=float, help="Sigma parameter for flow model")
    parser.add_argument('--save_model', action='store_true', help="Flag to save the trained model")
    parser.add_argument('--dataset', default='mnist', type=str, help="Dataset to train the model on")
    parser.add_argument('--log_interval', default=1, type=int, help="Logging interval for training")
    return parser.parse_args()


def get_model(args):
    if args.dataset == 'two_moons':
        return MLP(dim=2)
    else:  # MNIST
        return UNetModel(dim=(1, 28, 28), num_channels=32, num_res_blocks=1, num_classes=10)

def save_directory(args):
    if args.model_type == 'vfm' and args.learn_sigma:
        suffix = f"{args.model_type}_{args.loss_fn}_learned"
    elif args.model_type == 'vfm' and not args.learn_sigma:
        suffix = f"{args.model_type}_{args.loss_fn}_fixed"
    else:
        suffix = f"{args.model_type}"
    return os.path.join(os.getcwd(), f"results/{args.dataset}/{suffix}")

def main(args):
    savedir = save_directory(args)
    Path(savedir).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    flow_model = VFM(
        prior=StandardGaussianPrior(28**2) if args.dataset == 'mnist' else MultiGaussianPrior(2),
        variational_dist=GaussianVariationalDist(get_model(args)),
        interpolator=OTInterpolator(sigma_min=args.sigma),
    ).to(device)

    criterion = CRITERION_MAP[args.loss_fn]()
    optimizer = torch.optim.Adam(flow_model.variational_dist.posterior_mu_model.parameters(), lr=args.lr)

    num_params = sum(p.numel() for p in flow_model.variational_dist.posterior_mu_model.parameters())
    print(f"Number of parameters: {num_params}")
    print(f"Training parameters: {vars(args)}")

    dataloader = get_dataloader(args)
    train(dataloader, flow_model, criterion, optimizer, device, savedir, args)

    if args.save_model:
        torch.save(flow_model.state_dict(), f"{savedir}/model.pt")


def get_dataloader(args):
    if args.dataset == 'two_moons':
        return generate_two_moons(256000, args.batch_size)
    elif args.dataset == 'mnist':
        data = datasets.MNIST(
            "data",
            train=True,
            download=True,
            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]),
        )   
        return torch.utils.data.DataLoader(data, batch_size=args.batch_size, shuffle=True)
    else:
        raise ValueError("Invalid dataset argument")


def train(train_loader, model, criterion, optimizer, device, savedir, args):

    pbar = tqdm(total=args.num_epochs * len(train_loader))

    for epoch in range(args.num_epochs):
        for x_1 in train_loader:
            if isinstance(x_1, list):
                x_1 = x_1[0]

            optimizer.zero_grad()

            t, x_t = model.sample_t_and_x_t(x_1)
            t, x_t, x_1 = t.to(device), x_t.to(device), x_1.to(device)
            posterior = model.variational_dist(x_t, t)

            if args.dataset == 'mnist':
                x_1 = x_1.view(-1, 28*28)

            loss = criterion(posterior, x_1)
            loss.backward()
            optimizer.step()
            pbar.update(1)

        pbar.set_description(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
        if args.log_interval > 0 and (epoch+1) % args.log_interval == 0:
            evaluate(args, model, savedir, device, epoch+1)

    pbar.close()

    evaluate(args, model, savedir, device)


if __name__ == "__main__":
    args = get_args()
    main(args)