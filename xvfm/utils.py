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
from matplotlib.backends.backend_pdf import PdfPages


def generate(model, num_samples=100, steps=10, dev=None):

    if dev is None:
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    k = model.num_feat + sum(model.classes) + 1
    xt = torch.randn(num_samples, k, device=dev)
    trajectory = zeros((steps, xt.shape[0], xt.shape[1]), device=dev)

    with torch.no_grad():
        t_steps = torch.linspace(0, 1, steps, device=dev).unsqueeze(1)
        trajectory[0, :, :] = xt
        delta_t = 1.0 / steps
        for i in range(steps - 1):
            t = t_steps[i].expand(xt.shape[0], 1)
            mu = model(xt, t)
            v_t = (mu - (1 - 0.01) * xt) / (1 - (1 - 0.01) * t)
            # v_t = mu - (1 - 0.01) * xt
            xt += v_t * delta_t
            trajectory[i + 1, :, :] = xt

        return trajectory


def evaluate(args, model, test, dev, suffix):
    traj = generate(model, num_samples=test.shape[0], dev=dev)
    
    num = idx = args.num_feat
    k = num + len(args.classes) + 1 if sum(args.classes) != 0 else num + 1

    gen = zeros(test.shape[0], k, device=dev)
    gen[:, :num] = traj[-1, :, :num].to(torch.float)

    if sum(args.classes) != 0:
        for i, val in enumerate(args.classes):
            gen[:, num+i] =  torch.argmax(traj[0, :, idx:idx+val], dim=1)
            idx += val
    
    if args.task_type == "regression":
        gen[:, -1] = traj[-1, :, -1].to(torch.float)
    else:
        gen[:, -1] = torch.where(traj[-1, :, -1] > 0.5, 1, 0)

    if suffix % 50 == 0:
        scores = get_performance(gen, test, args, True)
    else:
        scores = get_performance(gen, test, args, False)

    if suffix % 100 == 0 or suffix == 1:
        plot_performance(test, gen, args, suffix)
        
    return scores


def plot_performance(test, gen, args, suffix):
    savedir = f'results/' + args.dataset
    num = args.num_feat
    k = num + len(args.classes) + 1 if sum(args.classes) != 0 else num + 1

    Path(savedir).mkdir(parents=True, exist_ok=True)

    data = test.cpu().numpy()
    synth = gen.cpu().numpy()
    grid_size = int(k**0.5) + (0 if (k**0.5).is_integer() else 1)

    with PdfPages(Path(savedir + f'/results_{suffix}.pdf')) as pdf:
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
        axes = axes.flatten()

        for i in range(num):
            df = pd.DataFrame({f"col_{i}": data[:, i], f"col_{i}_synth": synth[:, i]})
            sns.kdeplot(data=df, x=f"col_{i}", ax=axes[i])
            sns.kdeplot(data=df, x=f"col_{i}_synth", ax=axes[i])

        for i in range(len(args.classes)):
            min_val, max_val = synth[:, num+i].min(), synth[:, num+i].max()
            bins = np.arange(min_val - 0.5, max_val + 1.5, 1)

            sns.histplot(data[:, num+i], ax=axes[num+i], bins=bins, color="skyblue", edgecolor="black")
            sns.histplot(synth[:, num+i], ax=axes[num+i], bins=bins, color="purple", edgecolor="red")


        if args.task_type == "regression":
            sns.displot(data=data[:, -1], ax=axes[k-1])
            sns.displot(data=synth[:, -1], ax=axes[k-1])
        else:
            min_val, max_val = synth[:, -1].min(), synth[:, -1].max()
            bins = np.arange(min_val - 0.5, max_val + 1.5, 1)
            sns.histplot(data[:, -1], ax=axes[k-1], color="skyblue", edgecolor="black")
            sns.histplot(synth[:, -1], ax=axes[k-1], color="purple", edgecolor="red")

        axes[k-1].set_title(f"Column {k}")
        axes[k-1].set_xlabel("Value")
        axes[k-1].set_ylabel("Density")

        for j in range(k, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def clean_workspace(start_time):
    for root, dirs, files in os.walk(os.path.join(os.getcwd(), 'workspace')):
        for name in files:
            file_path = os.path.join(root, name)
            try:
                if name.endswith('.bkp') and os.path.getctime(file_path) > start_time:
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
