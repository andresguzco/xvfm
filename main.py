import os
import time
import torch
import wandb
import argparse

from xvfm import (
    VFM, 
    GaussianMultinomialPrior, 
    MultiMLP, 
    GaussianMultinomialDist, 
    OTInterpolator,
    evaluate, 
    clean_workspace
)
from pathlib import Path
from data import prepare_fast_dataloader, make_dataset, prepare_test_data


TABULAR = ['adult', 'beijing', 'default', 'magic', 'news', 'shoppers']


def GaussianMultinomial(posterior, x_1, y):
    k = posterior[0].mean.shape[1]
    llk = lambda dist, x: -1. * dist.log_prob(x).mean()
    a = llk(posterior[0], x_1[:, :k])
    if posterior[1] != False:
        b = sum([llk(posterior[1][i], x_1[:, k + i]) for i in range(len(posterior[1]))])
    else:
        b = 0
    c = llk(posterior[2], y.float())
    return a + b + c


def get_args():
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(description='TabVFM: Experiment')
    parser.add_argument('--num_epochs', default=100, type=int, help="Number of training epochs")
    parser.add_argument('--batch_size', default=4096, type=int, help="Training batch size")
    parser.add_argument('--lr', default=0.1, type=float, help="Learning rate for optimizer")
    parser.add_argument('--integration_steps', default=100, type=int, help="Number of steps for integration")
    parser.add_argument('--sigma', default=0.01, type=float, help="Sigma parameter for flow model")
    parser.add_argument('--dataset', default='adult', type=str, help="Dataset to train the model on")
    parser.add_argument('--log_interval', default=10, type=int, help="Logging interval for training")
    parser.add_argument('--seed', default=42, type=int, help="Random seed for reproducibility")
    parser.add_argument('--results_dir', default='results', type=str, help="Directory to save results")
    parser.add_argument('--d_layers', default=128, type=int, help="Hidden layer dimension.")
    parser.add_argument('--n_layers', default=3, type=int, help="Number of hidden layers for MLP")
    parser.add_argument('--dropout', default=0.1, type=float, help="Dropout rate for MLP")
    parser.add_argument('--dim_t', default=128, type=int, help="Dimension of time embedding")
    parser.add_argument('--data_path', default=None, type=str, help="Path to tabular dataset")
    parser.add_argument('--task_type', default=None, type=str, help="Downstream task to evaluate")
    return parser.parse_args()


def main(args):
    torch.manual_seed(args.seed)
    log = wandb.init(config=vars(args))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args.arch = [args.d_layers] * (args.n_layers)
    savedir = os.path.join(os.getcwd(), f"{args.results_dir}/{args.dataset}")
    # Path(savedir).mkdir(parents=True, exist_ok=True)

    if args.task_type == 'binclass':
        num_classes = 2
    else:
        num_classes = 0

    dataset = make_dataset(args.data_path, num_classes=num_classes)
    dataloader = prepare_fast_dataloader(dataset, split='train', batch_size=args.batch_size)

    test_data = prepare_test_data(dataset)
    
    args.num_feat = dataloader.num_feat
    args.classes = dataloader.classes
    args.d_in = sum(args.classes) + args.num_feat

    model = MultiMLP(
            args.d_in, 
            args.classes, 
            args.arch, 
            args.dropout, 
            args.dim_t, 
            args.num_feat, 
            args.task_type
            )
    prior = GaussianMultinomialPrior(args.num_feat, args.classes, args.task_type)
    variational = GaussianMultinomialDist(model, args.num_feat, args.classes, args.task_type)
    flow_model = VFM(prior, variational, OTInterpolator(args.sigma)).to(device)

    criterion = GaussianMultinomial
    params = flow_model.variational_dist.get_parameters()
    optimizer = torch.optim.Adam(params, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.01, total_iters=(args.num_epochs//10)
        )
    
    print(f"Number of parameters: {sum([p.numel() for p in params])}")
    print(f"Training parameters: {vars(args)}")

    train(dataloader, test_data, flow_model, criterion, optimizer, 
          scheduler, device, savedir, args, log)

    wandb.finish()


def train_step(model, optim, loss_fn, x1, y):
    optim.zero_grad()
    t, xt = model.sample_t_and_x_t(x1, y)
    posterior = model.variational_dist(xt, t, y)
    loss = loss_fn(posterior, x1, y)
    loss.backward()
    optim.step()
    return loss.item()


def train(train_loader, test, model, criterion, optimizer, 
          scheduler, device, savedir, args, wandb=None):
    
    if test.shape[0] > 5000:
        test_data = test[5000:, :].to(device)
    else:
        test_data = test.to(device)

    flag = False
    for epoch in range(args.num_epochs):
        start_time = time.time()
        epoch_loss = []
        
        for x_1, y in train_loader:
            x_1, y = x_1.to(device), y.view(-1, 1).to(device)
            epoch_loss.append(train_step(model, optimizer, criterion, x_1, y))

        scheduler.step()

        avg_loss = sum(epoch_loss) / len(epoch_loss)
        print(f"Epoch [{epoch+1}/{args.num_epochs}]: Loss: [{avg_loss:.4f}]", flush=True)

        if args.log_interval > 0 and (epoch + 1) % args.log_interval == 0:
            flag = True
        
        scores = evaluate(args, model, test_data, savedir, epoch + 1, flag, device)
        log_vals = scores | {'loss': avg_loss} 
        wandb.log(log_vals)
        clean_workspace(start_time)
        flag = False


if __name__ == "__main__":
    args = get_args()
    main(args)
