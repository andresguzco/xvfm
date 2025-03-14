import time
import torch
import wandb
import argparse
from tqdm import tqdm
from xvfm import Tabby, evaluate, clean_workspace, GuassianMultinomial
from data import prepare_fast_dataloader, make_dataset, prepare_test_data
from torch.nn.functional import one_hot


def get_args():
    parser = argparse.ArgumentParser(description='TabVFM')
    parser.add_argument('--dataset', default=None, type=str, help="Dataset to train the model on")
    parser.add_argument('--data_path', default=None, type=str, help="Path to tabular dataset")
    parser.add_argument('--task_type', default=None, type=str, help="Downstream task to evaluate")
    parser.add_argument('--epochs', default=8000, type=int, help="Number of training epochs")
    parser.add_argument('--batch_size', default=4096, type=int, help="Training batch size")
    parser.add_argument('--lr', default=1e-3, type=float, help="Learning rate for optimizer")
    parser.add_argument('--logging', default=100, type=int, help="Logging interval for training")
    parser.add_argument('--loss', default='llk', type=str, help="Loss function to use for training")
    parser.add_argument('--num_eval', default=1000, type=int, help="Number of synthetic observations")
    return parser.parse_args()


def main(args):
    torch.manual_seed(42)
    log = wandb.init(config=vars(args))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = make_dataset(args.data_path, num_classes=2 if args.task_type == "regression" else 0)
    dataloader = prepare_fast_dataloader(dataset, split='train', batch_size=args.batch_size)
    
    args.num_feat = dataloader.num_feat
    args.classes = dataloader.classes
    args.d_in = sum(args.classes) + args.num_feat

    if args.task_type == 'regression':
        args.d_in += 1
    else:
        args.d_in += 2

    model = Tabby(
        d_in=args.d_in, 
        num_feat=args.num_feat, 
        classes=args.classes, 
        task=args.task_type
        ).to(device)
    params = model.parameters()
    optimizer = torch.optim.Adam(params, lr=args.lr)
    criterion = GuassianMultinomial(
        num_feat=args.num_feat, classes=args.classes, task=args.task_type
        )
    
    print(f'Number of parameters: {sum([p.numel() for p in params])}')
    print(f'Training parameters: {vars(args)}')

    test_data = prepare_test_data(dataset)[:args.num_eval, :]

    for epoch in tqdm(range(args.epochs)):
        start_time = time.time()
        epoch_loss = 0.0
        num_batches = 0
        
        for x_1, y in dataloader:
            optimizer.zero_grad()
            x_1, y = x_1.to(device), y.view(-1, 1).to(device)

            x = torch.zeros((x_1.shape[0], args.d_in), device=device)
            x[:, :args.num_feat] = x_1[:, :args.num_feat]
            
            if sum(args.classes) != 0:
                idx = num = args.num_feat
                for i, val in enumerate(args.classes):
                    x[:, idx:idx+val] = one_hot(x_1[:, num+i].to(torch.int64), num_classes=val)
                    idx += val

            x0 = torch.randn((x_1.shape[0], args.d_in), device=device)
            t = torch.rand(x_1.shape[0], device=device).view(-1, 1)
            
            if args.task_type == 'regression':
                x[:, -1] = y.squeeze()
            else:
                x[:, -2:] = one_hot(y, num_classes=2).squeeze()

            xt = x * t + (1 - t) * x0 + torch.randn_like(x0) * 0.01

            res = model(xt, t)
                
            loss = criterion(res, torch.cat([x_1, y], dim=1), t)
            loss.backward()
            optimizer.step()

            epoch_loss  += loss
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        print(f'Epoch [{epoch+1}/{args.epochs}]: Loss: [{avg_loss:.4f}]', flush=True)

        if args.logging > 0 and ((epoch + 1) % args.logging == 0 or epoch == 0):
            scores = evaluate(args, model, test_data, device, epoch + 1)
        else:
            scores = {}
        
        log.log(scores | {'loss': avg_loss})
        clean_workspace(start_time)

    log.finish()


if __name__ == "__main__":
    args = get_args()
    main(args)
