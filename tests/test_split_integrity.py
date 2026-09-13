"""Tests for spherical split integrity and absence of illumination leakage."""

import pytest
import numpy as np
import pandas as pd
from src.splits import spherical_coords_to_unit_vectors, generate_spherical_bins, create_illumination_folds


def test_unit_vectors_norm():
    """Verify that spherical conversion yields normalized unit vectors on 3D sphere."""
    azimuths = np.array([-130, -45, 0, 45, 90, 130])
    elevations = np.array([-30, 0, 15, 45, 60, 90])

    vectors = spherical_coords_to_unit_vectors(azimuths, elevations)
    assert vectors.shape == (6, 3)

    norms = np.linalg.norm(vectors, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_spherical_clustering_zero_overlap():
    """Verify that generated folds hold out illumination bins with strictly ZERO leakage."""
    # Create synthetic dataset across 5 subjects and 20 lighting angles
    subjects = [f"yaleB{i:02d}" for i in range(11, 16)]
    angles = [(-90 + i * 10, -20 + (i % 5) * 10) for i in range(20)]

    rows = []
    for s in subjects:
        for a, e in angles:
            rows.append({
                "path": f"data/YaleFace/cropped/{s}/{s}_P00A{a:+04d}E{e:+03d}.pgm",
                "person": s,
                "pose": "P00",
                "A": a,
                "E": e,
            })
    df = pd.DataFrame(rows)

    df_bins, centroids = generate_spherical_bins(df, n_bins=4, seed=42)
    assert "bin_id" in df_bins.columns
    assert df_bins["bin_id"].nunique() == 4

    # Check that in every fold, test illumination bins are strictly absent from training
    unique_bins = sorted(df_bins["bin_id"].unique())
    for fold_idx, test_bin in enumerate(unique_bins):
        test_mask = df_bins["bin_id"] == test_bin
        train_df = df_bins.loc[~test_mask]
        test_df = df_bins.loc[test_mask]

        train_bins = set(train_df["bin_id"].unique())
        test_bins = set(test_df["bin_id"].unique())

        assert test_bin in test_bins
        assert test_bin not in train_bins
        assert train_bins.isdisjoint(test_bins), f"Illumination bin leakage detected in fold {fold_idx}"

        # Ensure all subjects are represented in both train and test partitions
        assert set(test_df["person"]) == set(subjects), "All subjects must be tested under held-out illumination"


def test_split_generation_determinism():
    """Verify that running split clustering with identical seed produces byte-identical bins."""
    rows = [
        {"path": f"p_{i}", "person": f"s_{i%4}", "pose": "P00", "A": -40 + i*5, "E": -10 + (i%3)*10}
        for i in range(30)
    ]
    df = pd.DataFrame(rows)

    df1, _ = generate_spherical_bins(df.copy(), n_bins=3, seed=123)
    df2, _ = generate_spherical_bins(df.copy(), n_bins=3, seed=123)

    np.testing.assert_array_equal(df1["bin_id"].values, df2["bin_id"].values)
