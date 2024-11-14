import os
import torch
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid

from xvfm.models.fm import CFM
from xvfm.models.unet import UNetModel

def main():
    savedir = "results/cond_mnist"
    os.makedirs(savedir, exist_ok=True)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    batch_size = 128
    n_epochs = 10

    trainset = datasets.MNIST(
        "data",
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]),
    )

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, drop_last=True
    )

    sigma = 0.0
    model = UNetModel(dim=(1, 28, 28), num_channels=32, num_res_blocks=1, num_classes=10, class_cond=True).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    flow_model = CFM(sigma=sigma)

    for epoch in range(n_epochs):
        for i, data in enumerate(train_loader):

            optimizer.zero_grad()

            x1 = data[0].to(device)
            y = data[1].to(device)
            x0 = torch.randn_like(x1)

            t, xt, ut = flow_model.sample_location_and_conditional_flow(x0, x1)
            vt = model(t, xt, y)
            loss = torch.mean((vt - ut) ** 2)

            loss.backward()
            optimizer.step()

            print(f"epoch: {epoch}, steps: {i}, loss: {loss.item():.4}", end="\r")

    with torch.no_grad():
        traj = flow_model.integration(
            torch.randn(100, 1, 28, 28, device=device),
            t_span=torch.linspace(0, 1, 2, device=device),
        )

    grid = make_grid(
        traj[-1, :100].view([-1, 1, 28, 28]).clip(-1, 1), value_range=(-1, 1), padding=0, nrow=10
    )
    img = ToPILImage()(grid)
    plt.imshow(img)
    plt.show()


if __name__ == "__main__":
    main()