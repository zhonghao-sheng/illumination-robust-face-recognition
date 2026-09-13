"""Focused checks for preprocessing and path handling."""

import cv2
import numpy as np
import pytest

from src.data_loader import FaceDataset, encode_labels, image_path, prepare_gray


def test_gray_preprocessing_is_per_image():
    gradient = np.tile(np.arange(32, dtype=np.uint8) * 8, (32, 1))
    image = prepare_gray(gradient, 24, 20)
    assert image.shape == (24, 20)
    assert image.dtype == np.float32
    assert abs(float(image.mean())) < 1e-5
    assert abs(float(image.std()) - 1) < 1e-3


def test_label_vocabulary_comes_from_training_only():
    (train, test), classes = encode_labels(["a", "b", "a"], ["b"])
    assert classes == ["a", "b"]
    assert train.tolist() == [0, 1, 0]
    assert test.tolist() == [1]
    with pytest.raises(ValueError, match="unseen"):
        encode_labels(["a"], ["b"])


def test_paths_stay_inside_dataset(tmp_path):
    root = tmp_path / "faces"
    root.mkdir()
    assert image_path(root, r"group\image.pgm") == root / "group" / "image.pgm"
    with pytest.raises(ValueError, match="leaves"):
        image_path(root, "../outside.pgm")


def test_efficientnet_augmentation_only_on_training_data(tmp_path):
    import pandas as pd
    root = tmp_path / "faces"
    root.mkdir()
    image = np.tile(np.arange(32, dtype=np.uint8) * 8, (32, 1))
    assert cv2.imwrite(str(root / "sample.pgm"), image)
    frame = pd.DataFrame({"path": ["sample.pgm"]})
    evaluation = FaceDataset(frame, np.array([0]), root, "efficientnet", training=False)
    first, _ = evaluation[0]
    second, _ = evaluation[0]
    assert first.shape == (3, 224, 224)
    assert (first == second).all()
    training = FaceDataset(frame, np.array([0]), root, "efficientnet", training=True, seed=42)
    augmented, _ = training[0]
    assert not (first == augmented).all()
