import torch
from sklearn.datasets import make_moons

def generate_two_moons(n_samples=50000, batch_size=256):
    train_dataset, _ = make_moons(n_samples=n_samples)
    train_dataset = torch.tensor(train_dataset, dtype=torch.float32)
    train_loader =torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    return train_loader