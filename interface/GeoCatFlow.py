import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from Engine import * 
from pathlib import Path
from torchvision import datasets, transforms
from torch.autograd import Function
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

class CustomLoss(Function):
    @staticmethod
    def forward(ctx, logits, labels):
        ctx.save_for_backward(logits, labels)
        log_probs = F.log_softmax(logits, dim=-1)
        loss = F.nll_loss(log_probs, labels)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        logits, labels = ctx.saved_tensors
        probs = F.softmax(logits, dim=-1)
        labels_one_hot = torch.zeros_like(probs)
        labels_one_hot.scatter_(1, labels.unsqueeze(1), 1)
        grad_logits = probs - labels_one_hot
        # grad_logits *= grad_output
        return grad_logits, None 


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
    savedir = os.path.join(os.getcwd(), "Results/GeoCatFlow")
    Path(savedir).mkdir(parents=True, exist_ok=True)

    # dim = 2
    patience = 500
    counter = 0
    best_loss = 1e10
    model = MLPS()

    optimizer = torch.optim.Adam(model.parameters())
    criterion = CustomLoss.apply

    # start = time.time()

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
            else:
                counter += 1
                if counter > patience:
                    break      

            loss.backward()
            optimizer.step()
            break 

        # if (k + 1) % 1000 == 0:
            # end = time.time()
            # print(f"{k+1}: Loss [{loss.item():0.6f}]. Time {(end - start):0.2f}")
            # start = end

            # model.eval()
            # correct = 0
            # total = 0

            # with torch.no_grad():
            #     for images, labels in test_loader:
            #         images = images.view(images.size(0), -1)
            #         outputs = model(images)
            #         _, predicted = torch.max(outputs.data, 1)
            #         total += labels.size(0)
            #         correct += (predicted == labels).sum().item()

            # accuracy = 100 * correct / total
            # print(f"Accuracy after {k+1} iterations: {accuracy:.2f}%")

    # print(f"{best_k}: Loss [{best_loss:0.6f}]. Accuracy: [{best_accuracy:.3f}]. Time [{(end - start):0.2f}].")
    torch.save(model, f"{savedir}/GeoCatFlow.pt")
    return best_loss, best_accuracy, best_k


if __name__ == "__main__":
    main()