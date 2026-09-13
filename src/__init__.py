"""Illumination-shift face recognition experiment components."""

from .data_loader import FaceDataset, encode_labels, load_classical_features
from .evaluate import evaluate_predictions
from .models import IlluminationCNN, LdaSvmClassifier, build_efficientnet_b0
from .splits import generate_spherical_bins, parse_dataset_metadata
from .train import predict_model, seed_everything, train_fixed_epochs

__all__ = [
    "FaceDataset",
    "encode_labels",
    "load_classical_features",
    "evaluate_predictions",
    "IlluminationCNN",
    "LdaSvmClassifier",
    "build_efficientnet_b0",
    "generate_spherical_bins",
    "parse_dataset_metadata",
    "predict_model",
    "seed_everything",
    "train_fixed_epochs",
]
