"""Grouped model selection keeps validation lighting out of training."""

import numpy as np
import pandas as pd
import pytest

from run_experiments import inner_partitions, load_fold


def test_inner_partitions_hold_out_complete_bins():
    frame = pd.DataFrame({"bin_id": np.repeat(np.arange(6), 3)})
    for train_indices, validation_indices in inner_partitions(frame):
        train_bins = set(frame.iloc[train_indices]["bin_id"])
        validation_bins = set(frame.iloc[validation_indices]["bin_id"])
        assert train_bins.isdisjoint(validation_bins)


def test_inner_partitions_require_enough_bins():
    frame = pd.DataFrame({"bin_id": [0, 0, 1, 1]})
    with pytest.raises(ValueError, match="at least"):
        list(inner_partitions(frame))


def test_outer_fold_rejects_overlapping_bins(tmp_path):
    train = pd.DataFrame({"path": ["a.pgm"], "person": ["a"], "bin_id": [0]})
    test = pd.DataFrame({"path": ["b.pgm"], "person": ["a"], "bin_id": [0]})
    train.to_csv(tmp_path / "fold_0_train.csv", index=False)
    test.to_csv(tmp_path / "fold_0_test.csv", index=False)
    with pytest.raises(ValueError, match="overlap"):
        load_fold(tmp_path, 0)


def test_outer_fold_rejects_overlapping_images(tmp_path):
    train = pd.DataFrame({"path": ["same.pgm"], "person": ["a"], "bin_id": [0]})
    test = pd.DataFrame({"path": ["same.pgm"], "person": ["a"], "bin_id": [1]})
    train.to_csv(tmp_path / "fold_0_train.csv", index=False)
    test.to_csv(tmp_path / "fold_0_test.csv", index=False)
    with pytest.raises(ValueError, match="image paths overlap"):
        load_fold(tmp_path, 0)
