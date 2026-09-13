"""Lightweight, data-free check of the linear and CNN model paths."""

import numpy as np
import torch

from src.models import IlluminationCNN, LdaSvmClassifier


def main():
    rng = np.random.default_rng(42)
    images = np.concatenate([
        rng.normal(label, 0.2, size=(6, 24, 20)) for label in range(3)
    ]).astype(np.float32)
    labels = np.repeat(np.arange(3), 6)
    baseline = LdaSvmClassifier(C=0.01).fit(images, labels)
    assert baseline.predict(images).shape == labels.shape
    cnn = IlluminationCNN(n_classes=3, kernel_size=5)
    logits = cnn(torch.randn(2, 1, 24, 20))
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()
    assert cnn.features[0].weight.grad is not None
    print("Linear baseline and CNN smoke checks passed.")


if __name__ == "__main__":
    main()
