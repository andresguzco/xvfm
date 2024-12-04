import torch
import ignite
import matplotlib.pyplot as plt

from fid import FIDNet

from ignite.metrics import FID
from torchvision.utils import make_grid
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Normalize, ToTensor, ToPILImage

def evaluate(args, model, savedir, plot: bool, device=None, suffix: str = None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    traj = model.generate(
        num_samples=1024 if args.dataset == 'two_moons' else 100, 
        steps=args.integration_steps, 
        device=device,
        method=args.int_method
    )

    if args.dataset == 'two_moons' and plot is True:
        n = 2000
        traj = traj.cpu().numpy()
        plt.figure(figsize=(6, 6))
        plt.scatter(traj[0, :n, 0], traj[0, :n, 1], s=10, alpha=0.8, c="black")
        plt.scatter(traj[:-2, :n, 0], traj[:-2, :n, 1], s=0.2, alpha=0.2, c="olive")
        plt.scatter(traj[-2, :n, 0], traj[-2, :n, 1], s=4, alpha=1, c="blue")
        plt.legend(["Prior sample z(S)", "Flow", "z(0)"])
        plt.xticks([])
        plt.yticks([])   
        filename = f"{savedir}/iter_{suffix}.png" if suffix else f"{savedir}/final.png"
        plt.savefig(filename)
        plt.close()
        return None

    else:
        real_dataset = MNIST(
            root='data', train=False, download=True, 
            transform=Compose([ToTensor(), Normalize((0.5,), (0.5,))])
            )
        real_loader = torch.utils.data.DataLoader(real_dataset, batch_size=100, shuffle=True)
        real_images = next(iter(real_loader))[0].to(device)[:100]
        generated_images = traj[-2, :100].view([-1, 1, 28, 28]).clip(-1, 1).to(device)

        evaluator = FIDNet()
        evaluator.load_state_dict(torch.load("checkpoints/fid_model.pt"))
        m = FID(num_features=10, device=device, feature_extractor=evaluator)
        m.update((real_images, generated_images))
        fid = m.compute()

        if plot:
            grid = make_grid(generated_images, value_range=(-1, 1), padding=0, nrow=10)
            img = ToPILImage()(grid)

            plt.imshow(img)
            plt.title(f"FID: {fid:.2f}")
            filename = f"{savedir}/iter_{suffix}.png" if suffix else f"{savedir}/final.png"
            plt.savefig(filename)
            plt.close()

        return fid