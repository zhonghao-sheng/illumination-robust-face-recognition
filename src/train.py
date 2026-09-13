"""Fixed-budget neural training and held-out prediction.

Hyperparameters are chosen on inner training-side folds. The final model is then
trained on the entire outer-training partition for a fixed number of epochs.
"""

import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_fixed_epochs(
    model: nn.Module, dataset, *, epochs: int, batch_size: int,
    learning_rate: float, weight_decay: float, device: torch.device, seed: int,
) -> nn.Module:
    if epochs < 1 or len(dataset) < 1:
        raise ValueError("Training requires images and at least one epoch")
    seed_everything(seed)
    model = model.to(device)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator, num_workers=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    for _ in range(epochs):
        model.train()
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(images), targets)
            loss.backward()
            optimizer.step()
    return model


def predict_model(model: nn.Module, dataset, *, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    predictions = []
    with torch.no_grad():
        for images, _ in loader:
            predictions.append(model(images.to(device)).argmax(dim=1).cpu().numpy())
    if not predictions:
        raise ValueError("Evaluation set is empty")
    return np.concatenate(predictions)
