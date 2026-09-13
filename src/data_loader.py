"""Image loading and model-specific, training-safe preprocessing."""

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def encode_labels(train: Sequence[str], *evaluation: Sequence[str]):
    """Create the identity vocabulary from training labels only."""
    classes = sorted(set(str(value) for value in train))
    mapping = {label: index for index, label in enumerate(classes)}
    encoded = []
    for values in (train, *evaluation):
        values = [str(value) for value in values]
        missing = set(values) - set(mapping)
        if missing:
            raise ValueError(f"Evaluation contains unseen identities: {sorted(missing)}")
        encoded.append(np.asarray([mapping[value] for value in values], dtype=np.int64))
    return encoded, classes


def image_path(data_root: str | Path, relative_path: str) -> Path:
    """Resolve a generated split path without accepting paths outside the data root."""
    root = Path(data_root).resolve()
    path = (root / relative_path.replace("\\", "/")).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Image path leaves the data directory: {relative_path}")
    return path


def read_gray(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Image is missing: {path}")
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    return image


def verify_image_paths(frame: pd.DataFrame, data_root: str | Path) -> None:
    """Fail before model fitting if a generated split references a missing image."""
    for relative_path in frame["path"]:
        path = image_path(data_root, relative_path)
        if not path.is_file():
            raise FileNotFoundError(f"Image is missing: {path}")


def prepare_gray(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """CLAHE and per-image standardization for the SVM and custom CNN."""
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    image = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(image)
    values = image.astype(np.float32) / 255.0
    return (values - values.mean()) / (values.std() + 1e-6)


def load_classical_features(
    frame: pd.DataFrame, data_root: str | Path, height: int, width: int
) -> np.ndarray:
    return np.stack([
        prepare_gray(read_gray(image_path(data_root, path)), height, width)
        for path in frame["path"]
    ]).astype(np.float32)


class FaceDataset(Dataset):
    """Read images lazily so 224-pixel RGB tensors need not reside in memory."""

    def __init__(
        self, frame: pd.DataFrame, labels: np.ndarray, data_root: str | Path,
        model: str, height: int = 96, width: int = 84,
        training: bool = False, seed: int = 42,
    ):
        if len(frame) != len(labels):
            raise ValueError("Image and label counts differ")
        if model not in ("cnn", "efficientnet"):
            raise ValueError("Unknown image model")
        self.paths = [image_path(data_root, path) for path in frame["path"]]
        self.labels = np.asarray(labels, dtype=np.int64)
        self.model = model
        self.height = 224 if model == "efficientnet" else height
        self.width = 224 if model == "efficientnet" else width
        self.training = training
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        gray = read_gray(self.paths[index])
        if self.model == "cnn":
            values = prepare_gray(gray, self.height, self.width)
            tensor = torch.from_numpy(values[None, :, :].copy())
        else:
            rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
            if self.training:
                angle = float(self.rng.uniform(-10, 10))
                matrix = cv2.getRotationMatrix2D((112, 112), angle, 1.0)
                rgb = cv2.warpAffine(rgb, matrix, (224, 224), borderMode=cv2.BORDER_REFLECT_101)
                contrast = float(self.rng.uniform(0.9, 1.1))
                brightness = float(self.rng.uniform(-10, 10))
                noise = self.rng.normal(0, 3, rgb.shape)
                rgb = np.clip(rgb.astype(np.float32) * contrast + brightness + noise, 0, 255).astype(np.uint8)
            tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255.0
            mean = torch.tensor((0.485, 0.456, 0.406))[:, None, None]
            std = torch.tensor((0.229, 0.224, 0.225))[:, None, None]
            tensor = (tensor - mean) / std
        return tensor, int(self.labels[index])
