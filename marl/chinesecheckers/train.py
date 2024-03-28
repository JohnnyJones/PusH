import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from torch.utils.data import DataLoader

def training_loop(model: nn.Module, dataloader: DataLoader, optim: torch.optim.Optimizer, loss_fn: nn.Module, epochs: int = 100, device=torch.device('cpu')):
    losses = []

    model.to(device)
    model.train()

    for epoch in tqdm(range(epochs)):
        for X, y_truth, y_value in dataloader:
            epoch_losses = []
            epoch_accuracies = []
            X, y_truth, y_value = X.to(device), y_truth.to(device), torch.tensor(y_value, dtype=torch.float32, device=device)
            pred_value, pred_truth = model(X)
            loss: torch.Tensor = loss_fn(pred_value, pred_truth, y_value, y_truth, model.parameters())
            optim.zero_grad()
            loss.backward()
            optim.step()
            epoch_losses.append(loss.item())
        losses.append(np.mean(epoch_losses))
    
    return losses