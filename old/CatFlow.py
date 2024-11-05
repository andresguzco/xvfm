import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from Engine import * 
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def binarize(image):
    return (image > 0.5).float()  

transform = transforms.Compose([
    transforms.ToTensor(),      
    transforms.Lambda(binarize) 
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

class MLPS(nn.Module):
    def __init__(self):
        super(MLPS, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def main():
    savedir = os.path.join(os.getcwd(), "Results/CatFlow")
    Path(savedir).mkdir(parents=True, exist_ok=True)

    # dim = 2
    patience = 500
    counter = 0
    best_loss = 1e10
    model = MLPS()

    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()


    for k in range(5000):
        model.train()
        optimizer.zero_grad()

        for images, labels in train_loader:
            images = images.view(images.size(0), -1)
            labels = labels  

            logits = model(images)
            loss = criterion(logits, labels)
            if loss.item() < best_loss:
                best_loss = loss.item()
                model.eval()
                total = 0
                correct = 0
                with torch.no_grad():
                    for images, labels in test_loader:
                        images = images.view(images.size(0), -1)
                        outputs = model(images)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()

                best_accuracy = 100 * correct / total
                best_k = k+1
                counter = 0
            else:
                counter += 1
                if counter > patience:
                    break
                
            loss.backward()
            optimizer.step()
            break 

    torch.save(model, f"{savedir}/CatFlow.pt")
    return best_loss, best_accuracy, best_k

if __name__ == "__main__":
    main()