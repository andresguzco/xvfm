import numpy as np
from tqdm import tqdm
import GeoCatFlow as script

best_epochs = []
best_losses = []
best_FDs = []

num_runs = 100
for i in tqdm(range(num_runs)):
    best_loss, best_FD, best_k = script.main()
    best_epochs.append(best_k)
    best_losses.append(best_loss)
    best_FDs.append(best_FD)

print(f"Best epoch average: {np.mean(best_epochs)}")
print(f"Best loss average: {np.mean(best_losses)}")
print(f"Best FD average: {np.mean(best_FDs)}")