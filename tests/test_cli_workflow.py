"""Exercise split generation and the public linear-model runner on PGM fixtures."""

import json
import subprocess
import sys

import cv2
import numpy as np
import pytest


@pytest.fixture
def fixture_dataset(tmp_path):
    root = tmp_path / "faces"
    root.mkdir()
    angles = [(-100, 0), (-90, 10), (-50, 0), (-40, 10),
              (0, 0), (10, 10), (50, 0), (60, 10), (100, 0), (110, 10)]
    rng = np.random.default_rng(42)
    for identity in range(3):
        folder = root / f"yaleB{identity + 11}"
        folder.mkdir()
        base = rng.integers(30, 220, size=(32, 32), dtype=np.uint8)
        for azimuth, elevation in angles:
            name = f"{folder.name}_P00A{azimuth:+04d}E{elevation:+03d}.pgm"
            variation = rng.integers(-30, 31, size=base.shape)
            image = np.clip(base.astype(np.int16) + variation, 0, 255).astype(np.uint8)
            assert cv2.imwrite(str(folder / name), image)
    return root


def test_public_cli_runs_grouped_nested_svm(fixture_dataset, tmp_path):
    splits = tmp_path / "splits"
    output = tmp_path / "run.json"
    generate = subprocess.run(
        [sys.executable, "generate_splits.py", "--data-dir", str(fixture_dataset),
         "--output-dir", str(splits), "--n-bins", "4"],
        capture_output=True, text=True,
    )
    assert generate.returncode == 0, generate.stderr
    run = subprocess.run(
        [sys.executable, "run_experiments.py", "--data-root", str(fixture_dataset),
         "--splits-dir", str(splits), "--folds", "0", "--models", "svm",
         "--svm-height", "16", "--svm-width", "16", "--output", str(output)],
        capture_output=True, text=True,
    )
    assert run.returncode == 0, run.stderr
    result = json.loads(output.read_text())
    assert len(result["folds"]) == 1
    assert "macro_f1" in result["folds"][0]["metrics"]["svm"]
    assert len(result["folds"][0]["selected"]["svm"]["inner_macro_f1"]) == 3


def test_missing_image_fails_clearly(fixture_dataset, tmp_path):
    splits = tmp_path / "splits"
    subprocess.run(
        [sys.executable, "generate_splits.py", "--data-dir", str(fixture_dataset),
         "--output-dir", str(splits), "--n-bins", "4"],
        check=True, capture_output=True, text=True,
    )
    next(fixture_dataset.glob("*/*.pgm")).unlink()
    run = subprocess.run(
        [sys.executable, "run_experiments.py", "--data-root", str(fixture_dataset),
         "--splits-dir", str(splits), "--folds", "0", "--models", "svm",
         "--svm-height", "16", "--svm-width", "16"],
        capture_output=True, text=True,
    )
    assert run.returncode != 0
    assert "Image is missing" in run.stderr
