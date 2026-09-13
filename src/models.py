"""The three model families compared in the illumination study."""

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from torch import nn


class LdaSvmClassifier(BaseEstimator, ClassifierMixin):
    """Fit LDA and a linear SVM together, entirely within a training fold."""

    def __init__(self, C: float = 0.01):
        self.C = C

    def fit(self, images: np.ndarray, labels: np.ndarray):
        features = images.reshape(len(images), -1)
        n_classes = len(np.unique(labels))
        if n_classes < 2:
            raise ValueError("LDA requires at least two training identities")
        self.pipeline_ = make_pipeline(
            LinearDiscriminantAnalysis(n_components=n_classes - 1),
            SVC(C=self.C, kernel="linear"),
        )
        self.pipeline_.fit(features, labels)
        return self

    def predict(self, images: np.ndarray) -> np.ndarray:
        return self.pipeline_.predict(images.reshape(len(images), -1))


class IlluminationCNN(nn.Module):
    """Three convolutional blocks with 32, 64 and 128 channels."""

    def __init__(self, n_classes: int, kernel_size: int = 5):
        super().__init__()
        if kernel_size not in (3, 5, 7):
            raise ValueError("kernel_size must be 3, 5, or 7")
        layers = []
        channels = 1
        for next_channels in (32, 64, 128):
            layers.extend((
                nn.Conv2d(channels, next_channels, kernel_size, padding=kernel_size // 2),
                nn.ReLU(),
                nn.MaxPool2d(2),
            ))
            channels = next_channels
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(128, n_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(images)).flatten(1))


def build_efficientnet_b0(n_classes: int, pretrained: bool = True) -> nn.Module:
    """Replace the ImageNet model's final classifier for closed-set identification."""
    from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = efficientnet_b0(weights=weights)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, n_classes)
    return model
