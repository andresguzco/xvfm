import math
import torch
import torch.nn as nn
import torch.distributions as dist
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import energy_distance
from scipy.linalg import sqrtm
from torchdyn.datasets import generate_moons
from scipy.spatial.distance import directed_hausdorff



def eight_normal_sample(n, dim, scale=1, var=1):
    m = torch.distributions.multivariate_normal.MultivariateNormal(
        torch.zeros(dim), math.sqrt(var) * torch.eye(dim)
    )
    centers = [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
        (1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
        (-1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
        (-1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
    ]
    centers = torch.tensor(centers) * scale
    noise = m.sample((n,))
    multi = torch.multinomial(torch.ones(8), n, replacement=True)
    data = []
    for i in range(n):
        data.append(centers[multi[i]] + noise[i])
    data = torch.stack(data)
    return data


def sample_moons(n, noise=0.2):
    x0, _ = generate_moons(n, noise=noise)
    return x0 * 3 - 1


def sample_8gaussians(n):
    return eight_normal_sample(n, 2, scale=5, var=0.1).float()


def plot_trajectories(traj, output=None):
    n = 2000
    plt.figure(figsize=(6, 6))
    plt.scatter(traj[0, :n, 0], traj[0, :n, 1], s=10, alpha=0.8, c="black")
    plt.scatter(traj[:, :n, 0], traj[:, :n, 1], s=0.2, alpha=0.2, c="olive")
    plt.scatter(traj[-1, :n, 0], traj[-1, :n, 1], s=4, alpha=1, c="blue")
    plt.legend(["Prior sample z(S)", "Flow", "z(0)"])
    plt.xticks([])
    plt.yticks([])

    if output is not None:
        plt.savefig(output)
        plt.close()
    else:
        plt.show()


def evaluate(sample_1, sample_2):
    sample_1 = sample_1.detach().cpu().numpy()
    sample_2 = sample_2.detach().cpu().numpy()

    mu1, mu2 = np.mean(sample_1, axis=0), np.mean(sample_2, axis=0)
    sigma1, sigma2 = np.cov(sample_1, rowvar=False), np.cov(sample_2, rowvar=False)    
    covmean = sqrtm(sigma1.dot(sigma2))
    
    if np.iscomplexobj(covmean):
            covmean = covmean.real

    frechet_distance = np.sum((mu1 - mu2)**2) + np.trace(sigma1 + sigma2 - 2 * covmean)
    
    # print(f"Fréchet Distance: {frechet_distance:.4f}.\n")
    return frechet_distance
