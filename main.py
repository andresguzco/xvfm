import time
import torch
import wandb
import argparse
from xvfm import Tabformer, evaluate, clean_workspace
from data import prepare_fast_dataloader, make_dataset, prepare_test_data
from torch.nn.functional import one_hot


class CustomLoss(torch.nn.Module):
    def __init__(self, num_feat, classes, task_type):
        super(CustomLoss, self).__init__()
        self.num_feat = num_feat
        self.classes = classes
        self.task_type = task_type
        self.mse = torch.nn.MSELoss(reduction='sum')
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, res, x1, _):
        a = 0
        for i in range(self.num_feat):
            a += self.mse(res[:, i], x1[:, i])

        b = 0
        idx = self.num_feat
        for val in self.classes:
            b += self.ce(res[:, idx:idx+val].float(), x1[:, idx:idx+val].float())
            idx += val

        if self.task_type == 'regression':
            c = self.mse(res[:, -1], x1[:, -1])
        else:
            c = self.ce(res[:, -1].float(), x1[:, -1].float())

        return a, b, c


class GuassianMultinomial(torch.nn.Module):
    def __init__(self, num_feat, classes, task_type):
        super(GuassianMultinomial, self).__init__()
        self.num_feat = num_feat
        self.classes = classes
        self.task_type = task_type

    def forward(self, res, x, t):
        llk = lambda dist, x: -1. * dist.log_prob(x).mean()

        # sigma2 = (torch.ones_like(t) - t).pow(2).view(-1, 1)

        mu = res[:, :self.num_feat]
        identity = torch.eye(mu.size(1)).to(mu.device).unsqueeze(0).expand(mu.size(0), -1, -1)
        scale = (1 - (1 - 0.01)* t.unsqueeze(1)**2) 
        sigma = scale * identity
        
        normal = torch.distributions.MultivariateNormal(mu, sigma)
        a = llk(normal, x[:, :self.num_feat])

        # a = 0
        # for i in range(self.num_feat):
        #     gauss = torch.distributions.Normal(res[:, i], 0.01)
        #     a += llk(gauss, x[:, i])

        b = 0
        if sum(self.classes) > 0:
            idx = self.num_feat
            for i, val in enumerate(self.classes):
                logits_pred = res[:, idx:idx+val]
                cat = torch.distributions.Categorical(logits_pred)
                b += llk(cat, x[:, self.num_feat + i].to(torch.int64))
                idx += val

        if self.task_type == 'regression':
            target = torch.distributions.Normal(res[:, -1], 0.01)
        else:
            target = torch.distributions.Bernoulli(res[:, -1].view(-1, 1))

        c = llk(target, x[:, -1].view(-1, 1))

        return a, b, c


def get_args():
    parser = argparse.ArgumentParser(description='TabVFM')
    parser.add_argument('--dataset', default=None, type=str, help="Dataset to train the model on")
    parser.add_argument('--data_path', default=None, type=str, help="Path to tabular dataset")
    parser.add_argument('--task_type', default=None, type=str, help="Downstream task to evaluate")
    parser.add_argument('--epochs', default=8000, type=int, help="Number of training epochs")
    parser.add_argument('--batch_size', default=1024, type=int, help="Training batch size")
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
    args.d_in = sum(args.classes) + args.num_feat + 1

    model = Tabformer(args.d_in, args.classes, args.num_feat, args.task_type).to(device)
    params = model.parameters()
    optimizer = torch.optim.Adam(params, lr=args.lr)
    
    print(f"Number of parameters: {sum([p.numel() for p in params])}")
    print(f"Training parameters: {vars(args)}")
    
    if args.loss == 'llk':
        criterion = GuassianMultinomial(args.num_feat, args.classes, args.task_type)
    else:
        criterion = CustomLoss(args.num_feat, args.classes, args.task_type)

    k = args.num_feat + sum(args.classes) + 1

    test_data = prepare_test_data(dataset)[:args.num_eval, :]
    
    for epoch in range(args.epochs):
        start_time = time.time()
        epoch_loss = []
        
        for x_1, y in dataloader:

            optimizer.zero_grad()

            x_1, y = x_1.to(device), y.view(-1, 1).to(device)

            x0 = torch.randn((x_1.shape[0], k), device=device)
            t = torch.rand(x_1.shape[0], device=device).view(-1, 1)
            x_oh = torch.zeros((x_1.shape[0], k), device=device)
            x_oh[:, :args.num_feat] = x_1[:, :args.num_feat]
            
            if sum(args.classes) != 0:
                idx = num = args.num_feat
                for i, val in enumerate(args.classes):
                    x_oh[:, idx:idx+val] = one_hot(x_1[:, num+i].to(torch.int64), num_classes=val)
                    idx += val

            x_oh[:, -1] = y.squeeze()
            xt = x_oh * t + (1 - t) * x0 + torch.randn_like(x0) * 0.01
            res = model(xt, t)

            x_in = torch.cat([x_1, y], dim=1) if args.loss == 'llk' else torch.cat([x_oh, y], dim=1)
                
            loss_gauss, loss_cat, loss_target = criterion(res, x_in, t)
            loss = loss_gauss + loss_cat + loss_target

            loss.backward()
            optimizer.step()
            epoch_loss.append(loss)

        avg_loss = sum(epoch_loss) / len(epoch_loss)
        print(f"Epoch [{epoch+1}/{args.epochs}]: Loss: [{avg_loss:.4f}]", flush=True)

        if args.logging > 0 and ((epoch + 1) % args.logging == 0 or epoch == 0):
            scores = evaluate(args, model, test_data, device, epoch + 1)
        else:
            scores = {}
        
        log.log(scores | {'loss': avg_loss} )
        clean_workspace(start_time)

    log.finish()


if __name__ == "__main__":
    args = get_args()
    main(args)
