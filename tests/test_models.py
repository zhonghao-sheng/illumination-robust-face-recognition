"""Model-family checks without pretrained weight downloads."""

import numpy as np
import torch

from src.models import IlluminationCNN, LdaSvmClassifier, build_efficientnet_b0


def test_lda_svm_fits_within_training_data():
    rng = np.random.default_rng(42)
    images = np.concatenate([
        rng.normal(loc=label, scale=0.2, size=(8, 12, 10))
        for label in range(3)
    ]).astype(np.float32)
    labels = np.repeat(np.arange(3), 8)
    model = LdaSvmClassifier(C=0.01).fit(images, labels)
    predictions = model.predict(images)
    assert predictions.shape == labels.shape
    assert "lineardiscriminantanalysis" in model.pipeline_.named_steps
    assert "svc" in model.pipeline_.named_steps


def test_custom_cnn_backpropagates():
    model = IlluminationCNN(n_classes=4, kernel_size=5)
    logits = model(torch.randn(3, 1, 48, 40))
    assert logits.shape == (3, 4)
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 2])).backward()
    assert model.features[0].weight.grad is not None


def test_efficientnet_head_matches_subject_count():
    model = build_efficientnet_b0(n_classes=4, pretrained=False)
    assert model.classifier[-1].out_features == 4
