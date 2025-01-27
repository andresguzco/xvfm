import os
import torch
import shutil
import seaborn as sns
import matplotlib.pyplot as plt

from torch import zeros
from xvfm.eval import get_performance
from matplotlib.backends.backend_pdf import PdfPages
from torch.distributions import Categorical, Bernoulli


def evaluate(args, model, test, savedir, suffix, plotting, dev):
    traj = model.generate(num_samples=test.shape[0], steps=args.integration_steps, device=dev)
    
    num = cum_sum = args.num_feat
    if sum(args.classes) != 0:
        k = num + len(args.classes) + 1
    else:
        k = num + 1

    gen = zeros(test.shape[0], k, device=dev)
    gen[:, :num] = traj[-1, :, :num].to(torch.float)
    if sum(args.classes) != 0:
        for i, val in enumerate(args.classes):
            gen[:, num + i] =  Categorical(traj[-1, :, cum_sum: cum_sum + val]).sample().to(torch.int)
            cum_sum += val
    
    if args.task_type == "regression":
        gen[:, -1] = traj[-1, :, -1].to(torch.float)
    else:
        gen[:, -1] = Bernoulli(traj[-1, :, -1]).sample().to(torch.int)
    
    # if plotting:
    #     data = test.cpu().numpy()
    #     synth = gen.cpu().numpy()
    #     grid_size = int(k**0.5) + (0 if (k**0.5).is_integer() else 1)

    #     with PdfPages(savedir + f'/results_{suffix}.pdf') as pdf:
    #         fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
    #         axes = axes.flatten()

    #         for i in range(k):
    #             sns.histplot(data[:, i], kde=True, ax=axes[i], color="skyblue", edgecolor="black")
    #             sns.histplot(synth[:, i], kde=True, ax=axes[i], color="purple", edgecolor="red")
    #             axes[i].set_title(f"Column {i+1}")
    #             axes[i].set_xlabel("Value")
    #             axes[i].set_ylabel("Frequency")

    #         for j in range(k, len(axes)):
    #             axes[j].axis("off")

    #         plt.tight_layout()
    #         pdf.savefig(fig)
    #         plt.close(fig)

    scores = get_performance(gen, test, args, plotting)
    return scores



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
