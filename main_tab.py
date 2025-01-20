import os
import torch
import argparse
import wandb
import json
import numpy as np
import data.tabular as tab

from tqdm import tqdm
from pathlib import Path
from xvfm.flow import VFM
from xvfm.unet import UNetModel
from data.utils import evaluate
from xvfm.prior import StandardGaussianPrior, MultiGaussianPrior, GaussianMultinomialPrior
from xvfm.models import MLP, MultiMLP
from torchvision import datasets, transforms
from data.two_moons import generate_two_moons
from xvfm.variational import GaussianVariationalDist, GaussianMultinomialDist
from xvfm.interpolator import OTInterpolator
from torchvision.transforms import Compose, Normalize, ToTensor, ToPILImage


TABULAR = ['abalone', 'adult', 'buddy', 'california', 'cardio', 'churn2', 
           'diabetes', 'fb-comments', 'gesture', 'higgs-small', 'house', 
           'insurance', 'king', 'miniboone', 'wilt']


def MSE(posterior, x_1, _): 
    return torch.mean((x_1 - posterior.mean) ** 2)


def Gaussian(posterior, x_1, _):
    return -1. * posterior.log_prob(x_1).mean()


def GaussianMultinomial(posterior, x_1, n):
    k = x_1.shape[1] - len(n)
    a = -1. * posterior[0].log_prob(x_1[:, :k]).mean()
    b = sum([posterior[1][i].log_prob(x_1[:, k + i]).mean() for i, val in enumerate(n)]) * -1.
    return a + b


CRITERION_MAP = {
    'MSE': MSE,
    'Gaussian': Gaussian,
    'GaussianMultinomial': GaussianMultinomial
}


def get_args():
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(description='Two Moon Experiment')
    parser.add_argument('--num_epochs', default=1000, type=int, help="Number of training epochs")
    parser.add_argument('--batch_size', default=4096, type=int, help="Training batch size")
    parser.add_argument('--lr', default=1e-5, type=float, help="Learning rate for optimizer")
    parser.add_argument('--loss_fn', default='Gaussian', type=str, help="Loss function for VFM: 'MSE', 'SSM', or 'Gaussian'")
    parser.add_argument('--learn_sigma', type=bool, default=True, help="Flag to learn sigma in VFM")
    parser.add_argument('--learned_structure', default='scalar', help="Flag to learn structure in VFM")
    parser.add_argument('--int_method', default='euler', help="Integration method for trajectory plotting: 'euler', 'adaptive'")
    parser.add_argument('--integration_steps', default=1000, type=int, help="Number of steps for integration in trajectory plotting")
    parser.add_argument('--sigma', default=0.1, type=float, help="Sigma parameter for flow model")
    parser.add_argument('--save_model', action='store_true', help="Flag to save the trained model")
    parser.add_argument('--dataset', default='mnist', type=str, help="Dataset to train the model on")
    parser.add_argument('--log_interval', default=10, type=int, help="Logging interval for training")
    parser.add_argument('--seed', default=42, type=int, help="Random seed for reproducibility")
    parser.add_argument('--checkpoint_interval', default=100, type=int, help="Interval to save checkpoints")
    parser.add_argument('--checkpoint_dir', default='checkpoints', type=str, help="Directory to save checkpoints")
    parser.add_argument('--results_dir', default='results', type=str, help="Directory to save results")
    parser.add_argument('--d_layers', default=[256, 1024, 1024, 1024, 1024, 512], type=list, help="Hidden layers for MLP")
    parser.add_argument('--dropout', default=0.1, type=float, help="Dropout rate for MLP")
    parser.add_argument('--dim_t', default=128, type=int, help="Dimension of time embedding")
    parser.add_argument('--data_path', default=None, type=str, help="Path to tabular dataset")
    parser.add_argument('--num_classes', default=None, type=int, help="List of classes for tabular dataset")
    parser.add_argument('--transformation', default=None, type=str, help="Transformation to apply to tabular dataset")
    parser.add_argument('--is_y_cond', default=False, type=bool, help="Flag to condition on y in tabular dataset")
    return parser.parse_args()


def get_model(args):
    if args.dataset == 'two_moons':
        if args.learn_sigma and args.learned_structure == "scalar":
            return (MLP(dim=2), MLP(dim=0, out_dim=1))
        elif args.learn_sigma and args.learned_structure == "vector":
            return (MLP(dim=2), MLP(dim=0, out_dim=2))
        elif args.learn_sigma and args.learned_structure == "matrix":
            return (MLP(dim=2), MLP(dim=0, out_dim=2**2))
        else:
            return (MLP(dim=2), None)
    elif args.dataset == 'mnist':
        if args.learn_sigma and args.learned_structure == "scalar":
            return (
                UNetModel(dim=(1, 28, 28), num_channels=32, num_res_blocks=1, num_classes=10), 
                MLP(dim=0, out_dim=1)
            )
        elif args.learn_sigma and args.learned_structure == "vector":
            return (
                UNetModel(dim=(1, 28, 28), num_channels=32, num_res_blocks=1, num_classes=10), 
                MLP(dim=0, out_dim=28*28)
            )
        elif args.learn_sigma and args.learned_structure == "matrix":
            return (
                UNetModel(dim=(1, 28, 28), num_channels=32, num_res_blocks=1, num_classes=10), 
                MLP(dim=0, out_dim=(28*28)**2)
            )
        else:
            return (UNetModel(dim=(1, 28, 28), num_channels=32, num_res_blocks=1, num_classes=10), None)
    elif args.dataset in TABULAR:
        return MultiMLP(
            d_in=args.d_in, 
            classes=args.classes,
            d_layers=args.d_layers, 
            dropout=args.dropout, 
            is_y_cond=args.is_y_cond, 
            dim_t=args.dim_t,
            num_feat=args.num_feat
        )
    else:
        raise ValueError("Invalid dataset argument")

def get_directories(args):
    if args.loss_fn == 'Gaussian' and args.learn_sigma:
        suffix = f"{args.loss_fn}_learned_{args.learned_structure}"
    elif args.loss_fn == 'Gaussian' and not args.learn_sigma:
        suffix = f"{args.loss_fn}_fixed"
    else:
        suffix = f"{args.loss_fn}"
    return os.path.join(os.getcwd(), f"{args.results_dir}/{args.dataset}/{suffix}")

def main(args):

    torch.manual_seed(args.seed)
    savedir = get_directories(args)
    Path(savedir).mkdir(parents=True, exist_ok=True)
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    log = wandb.init(project="XVFM", config=vars(args))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataloader = get_dataloader(args)
    if args.dataset in TABULAR:
        assert args.data_path is not None, "Data path must be provided for tabular datasets"
        K = dataloader.classes
        if len(K) == 0:
            K = np.array([0])

        num_feat = dataloader.num_feat
        d_in = np.sum(K) + num_feat
        args.num_feat = num_feat
        args.classes = K
        args.d_in = d_in

    if args.dataset == 'mnist':
        prior = StandardGaussianPrior(28**2)
        variational = GaussianVariationalDist(*get_model(args))
    elif args.dataset == 'two_moons':
        prior = MultiGaussianPrior(2)
        variational = GaussianVariationalDist(*get_model(args))
    elif args.dataset in TABULAR:
        assert args.data_path is not None, "Data path must be provided for tabular datasets"
        prior = GaussianMultinomialPrior(num_feat=dataloader.num_feat, cat_feat=K)
        variational = GaussianMultinomialDist(get_model(args), num_feat=args.num_feat, cat_feat=K)
    else:
        raise ValueError("Invalid dataset argument")

    flow_model = VFM(prior=prior, variational_dist=variational, interpolator=OTInterpolator(sigma_min=args.sigma)).to(device)

    criterion = CRITERION_MAP[args.loss_fn]
    params = flow_model.variational_dist.get_parameters()
    optimizer = torch.optim.Adam(params, lr=args.lr)
    
    print(f"Number of parameters: {sum([p.numel() for p in params])}")
    print(f"Training parameters: {vars(args)}")

    train(dataloader, flow_model, criterion, optimizer, device, savedir, args, log)
    # train(dataloader, flow_model, criterion, optimizer, device, savedir, args)

    wandb.finish()

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
            transform=Compose([ToTensor(), Normalize((0.5,), (0.5,))])
        )   
        return torch.utils.data.DataLoader(data, batch_size=args.batch_size, shuffle=True)
    elif args.dataset in TABULAR:
        assert args.data_path is not None, "Data path must be provided for tabular datasets"
        assert args.num_classes is not None, "Classes must be provided for tabular datasets"
        T = tab.Transformations(args.transformation)
        dataset = tab.make_dataset(args.data_path, T, num_classes=args.num_classes, is_y_cond=args.is_y_cond)
        train_loader = tab.prepare_fast_dataloader(dataset, split='train', batch_size=args.batch_size)
        return train_loader
    else:
        raise ValueError("Invalid dataset argument")


def train(train_loader, model, criterion, optimizer, device, savedir, args, wandb=None):

    pbar = tqdm(total=args.num_epochs)
    if args.loss_fn == 'Gaussian' and args.learn_sigma:
        suffix = f"{args.loss_fn}_learned_{args.learned_structure}"
    elif args.loss_fn == 'Gaussian' and not args.learn_sigma:
        suffix = f"{args.loss_fn}_fixed"
    else:
        suffix = f"{args.loss_fn}"
    
    n = train_loader.classes

    for epoch in range(args.num_epochs):
        for x_1 in train_loader:
            if isinstance(x_1, list):
                x_1 = x_1[0]
                y = x_1[1]
            
            optimizer.zero_grad()

            t, x_t = model.sample_t_and_x_t(x_1)
            t, x_t, x_1 = t.to(device), x_t.to(device), x_1.to(device)
            posterior = model.variational_dist(x_t, t, y)

            if args.dataset == 'mnist':
                x_1 = x_1.view(-1, 28*28)
            
            loss = criterion(posterior, x_1, n)
            loss.backward()
            optimizer.step()

        if args.log_interval > 0 and (epoch + 1) % args.log_interval == 0:
            score = evaluate(args, model, savedir, device, epoch+1, x_1)

        if args.checkpoint_interval > 0 and (epoch + 1) % args.checkpoint_interval == 0:
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "loss": loss.item()
                }, 
                f"{args.checkpoint_dir}/{suffix}.pt")
        #
        # wandb.log({"loss": loss.item(), "fid": score})
        wandb.log({"loss": loss.item()})
        pbar.update(1)

    pbar.close()
    evaluate(args, model, savedir, device)


if __name__ == "__main__":
    args = get_args()
    main(args)
