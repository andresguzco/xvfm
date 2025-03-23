import os
import torch
import shutil
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path
from torch import zeros
from xvfm.eval import get_performance
from torchdiffeq import odeint_adjoint
from matplotlib.backends.backend_pdf import PdfPages


class Velocity(torch.nn.Module):
    def __init__(self, model):
        super(Velocity, self).__init__()
        self.model = model

    def forward(self, t, x):
        t = t * torch.ones(x.shape[0]).unsqueeze(1).to(x.device)
        mu = self.model(x, t)
        v_t = (mu - (1 - 0.01) * x) / (1 - (1 - 0.01) * t)
        return v_t


def generate(model, num_samples, dev):
    x0 = torch.randn(num_samples, model.d_in, device=dev)
    t = torch.tensor([0.0, 1.0]).to(dev)
    vf = Velocity(model)
    with torch.no_grad():
        trajectory = odeint_adjoint(vf, x0, t, method="dopri5", rtol=1e-5, atol=1e-5)

    return trajectory[1]


def evaluate(args, model, test, dev, suffix):
    num = idx = args.num_feat
    k = num + len(args.classes) + 1 if sum(args.classes) != 0 else num + 1

    traj = generate(model, num_samples=test.shape[0], dev=dev)
    gen = zeros(test.shape[0], k, device=dev)
    gen[:, :num] = traj[:, :num].to(torch.float)

    if sum(args.classes) != 0:
        for i, val in enumerate(args.classes):
            gen[:, num + i] = torch.argmax(traj[:, idx : idx + val], dim=1)
            idx += val

    if args.task_type == "regression":
        gen[:, -1] = traj[:, -1].to(torch.float)
    else:
        gen[:, -1] = torch.argmax(traj[:, -2:], dim=1)

    if suffix % 50 == 0 or suffix == 1:
        scores = get_performance(gen, test, args, True)
    else:
        scores = get_performance(gen, test, args, False)

    if suffix % 100 == 0 or suffix == 1:
        plot_performance(test, gen, args, suffix)

    return scores


def plot_performance(test, gen, args, suffix):
    savedir = f"results/" + args.dataset
    num = args.num_feat
    k = num + len(args.classes) + 1 if sum(args.classes) != 0 else num + 1

    Path(savedir).mkdir(parents=True, exist_ok=True)

    data = test.cpu().numpy()
    synth = gen.cpu().numpy()
    grid_size = int(k**0.5) + (0 if (k**0.5).is_integer() else 1)

    with PdfPages(Path(savedir + f"/results_{suffix}.pdf")) as pdf:
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
        axes = axes.flatten()

        for i in range(num):
            df = pd.DataFrame({f"col_{i}": data[:, i], f"col_{i}_synth": synth[:, i]})
            sns.kdeplot(data=df, x=f"col_{i}", ax=axes[i])
            sns.kdeplot(data=df, x=f"col_{i}_synth", ax=axes[i])

        for i in range(len(args.classes)):
            min_val, max_val = synth[:, num + i].min(), synth[:, num + i].max()
            bins = np.arange(min_val - 0.5, max_val + 1.5, 1)
            sns.histplot(
                data[:, num + i],
                ax=axes[num + i],
                bins=bins,
                color="skyblue",
                edgecolor="black",
            )
            sns.histplot(
                synth[:, num + i],
                ax=axes[num + i],
                bins=bins,
                color="purple",
                edgecolor="red",
            )

        if args.task_type == "regression":
            sns.displot(data=data[:, -1], ax=axes[k - 1])
            sns.displot(data=synth[:, -1], ax=axes[k - 1])
        else:
            min_val, max_val = synth[:, -1].min(), synth[:, -1].max()
            bins = np.arange(min_val - 0.5, max_val + 1.5, 1)
            sns.histplot(
                data[:, -1], ax=axes[k - 1], color="skyblue", edgecolor="black"
            )
            sns.histplot(synth[:, -1], ax=axes[k - 1], color="purple", edgecolor="red")

        axes[k - 1].set_title(f"Column {k}")
        axes[k - 1].set_xlabel("Value")
        axes[k - 1].set_ylabel("Density")

        for j in range(k, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def clean_workspace(start_time):
    for root, dirs, files in os.walk(os.path.join(os.getcwd(), "workspace")):
        for name in files:
            file_path = os.path.join(root, name)
            try:
                if name.endswith(".bkp") and os.path.getctime(file_path) > start_time:
                    os.remove(file_path)
            except:
                pass

        for name in dirs:
            dir_path = os.path.join(root, name)
            try:
                if os.path.getctime(dir_path) > start_time:
                    shutil.rmtree(dir_path)
            except:
                pass
