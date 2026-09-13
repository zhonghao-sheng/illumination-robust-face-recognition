"""Spherical illumination clustering and grouped cross-validation split generation.

Why standard k-fold cross-validation fails for illumination robustness:
In facial recognition, random i.i.d. splitting places the exact same lighting angles
in both training and test sets across different subjects. This evaluates recognition under
memorized lighting setups rather than out-of-distribution illumination shifts.

Algorithm:
1. Maps light direction angles (Azimuth, Elevation) onto 3D unit-sphere directional embeddings.
   Since ||u|| = ||v|| = 1, squared Euclidean distance ||u - v||^2 = 2 - 2(u . v) is monotonically
   related to cosine distance, clustering illumination directions by angular proximity.
2. Applies KMeans clustering to partition the unit sphere into K spatial regions.
3. Applies a best-effort greedy rebalancing heuristic across illumination directions to reduce
   cluster size disparity. Note: Because all images sharing a discrete (Azimuth, Elevation) pair
   must remain in the same bin to prevent angle leakage, bin sizes cannot be made strictly uniform;
   the actual sample count spread is documented rather than assuming an exact tolerance guarantee.
4. Generates cross-validation folds by holding out one spatial cluster per fold.
"""

import os
import re
import json
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Union
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


FILENAME_PATTERN = re.compile(
    r"^(yaleB\d+)_P(\d{2})A([+-]?\d+)E([+-]?\d+)\.pgm$", re.IGNORECASE
)


def spherical_coords_to_unit_vectors(
    azimuth_deg: np.ndarray,
    elevation_deg: np.ndarray,
    convention: str = "physical",
) -> np.ndarray:
    """Convert spherical angles (Azimuth, Elevation) in degrees to 3D Cartesian unit vectors.
    
    Conventions:
    - "physical" (default):
        Camera optical axis along +z, horizontal axis x, vertical elevation y.
        Azimuth A is horizontal angle; Elevation E is vertical angle.
        x = cos(E) * sin(A)
        y = sin(E)
        z = cos(E) * cos(A)
        
    - "legacy" (alternate angle convention):
        Inverts azimuth and elevation, treating azimuth A as the out-of-plane angle:
        x = cos(A) * cos(E)
        y = cos(A) * sin(E)
        z = sin(A)
        
    Args:
        azimuth_deg: 1D array of azimuth angles in degrees.
        elevation_deg: 1D array of elevation angles in degrees.
        convention: "physical" or "legacy".
        
    Returns:
        (N, 3) array of unit vectors [x, y, z] on the sphere surface.
    """
    rad = np.pi / 180.0
    a_rad = np.asarray(azimuth_deg, dtype=np.float64) * rad
    e_rad = np.asarray(elevation_deg, dtype=np.float64) * rad

    if convention == "physical":
        x = np.cos(e_rad) * np.sin(a_rad)
        y = np.sin(e_rad)
        z = np.cos(e_rad) * np.cos(a_rad)
    elif convention == "legacy":
        x = np.cos(a_rad) * np.cos(e_rad)
        y = np.cos(a_rad) * np.sin(e_rad)
        z = np.sin(a_rad)
    else:
        raise ValueError(f"Unknown spherical conversion convention: {convention}")

    return np.stack([x, y, z], axis=1)


def parse_dataset_metadata(data_dir: Union[str, Path]) -> pd.DataFrame:
    """Scan dataset directory and extract subject, pose, and illumination metadata.
    
    Stores clean relative paths relative to data_dir (e.g. "yaleB11/yaleB11_P00A+000E+00.pgm"),
    preventing path-duplication bugs when data_root is passed to the loader.
    
    Args:
        data_dir: Root path containing subject subfolders (e.g. data/YaleFace/cropped).
        
    Returns:
        DataFrame with columns ["path", "person", "pose", "A", "E"].
    """
    p_dir = Path(data_dir)
    if not p_dir.exists():
        raise FileNotFoundError(f"Dataset root directory does not exist: {p_dir}")

    rows = []
    for subject_folder in sorted(p_dir.iterdir()):
        if not subject_folder.is_dir():
            continue
        for img_path in sorted(subject_folder.glob("*.pgm")):
            fname = img_path.name
            if "ambient" in fname.lower():
                continue
            m = FILENAME_PATTERN.match(fname)
            if not m:
                continue

            person_id = m.group(1)
            pose_id = f"P{m.group(2)}"
            azimuth = int(m.group(3))
            elevation = int(m.group(4))

            # Store portable relative path strictly relative to data_dir
            rel_path = str(img_path.relative_to(p_dir)).replace("\\", "/")
            rows.append({
                "path": rel_path,
                "person": person_id,
                "pose": pose_id,
                "A": azimuth,
                "E": elevation,
            })

    if not rows:
        raise ValueError(f"No valid Yale B PGM images found in {p_dir}")

    return pd.DataFrame(rows)


def generate_spherical_bins(
    df: pd.DataFrame,
    n_bins: int = 10,
    seed: int = 42,
    do_balance: bool = True,
    convention: str = "physical",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Cluster unique (Azimuth, Elevation) pairs on 3D unit-sphere directional embeddings.
    
    Uses scikit-learn KMeans on unit vectors. A best-effort greedy heuristic moves entire
    (Azimuth, Elevation) clusters towards underrepresented spatial bins when distance permits.
    
    Args:
        df: Metadata dataframe with columns ["A", "E"].
        n_bins: Number of spherical illumination clusters.
        seed: Random seed for KMeans initialization.
        do_balance: If True, applies best-effort greedy balancing across direction groups.
        convention: "physical" or "legacy" angle projection.
        
    Returns:
        Tuple of:
        - df_with_bins: Original df enriched with "bin_id" column.
        - centroids: DataFrame of sorted cluster centroids ["x", "y", "z", "bin_id"].
    """
    if not {"A", "E"}.issubset(df.columns):
        raise ValueError("Dataframe must contain illumination angle columns A and E.")

    unique_dirs = df[["A", "E"]].drop_duplicates().reset_index(drop=True).copy()
    unit_vectors = spherical_coords_to_unit_vectors(
        unique_dirs["A"].values,
        unique_dirs["E"].values,
        convention=convention,
    )

    km = KMeans(n_clusters=n_bins, n_init=50, random_state=seed)
    raw_clusters = km.fit_predict(unit_vectors)

    # Sort cluster centroids deterministically by z, y, x
    centroids = pd.DataFrame(km.cluster_centers_, columns=["x", "y", "z"])
    centroids["old_id"] = range(n_bins)
    centroids = centroids.sort_values(["z", "y", "x"]).reset_index(drop=True)
    remap = {row["old_id"]: idx for idx, row in centroids.iterrows()}
    centroids["bin_id"] = range(n_bins)
    centroids = centroids.drop(columns=["old_id"])

    unique_dirs["bin_id"] = [remap[c] for c in raw_clusters]
    df_out = df.merge(unique_dirs, on=["A", "E"], how="left")

    if do_balance and n_bins > 1:
        dir_counts = df_out.groupby(["A", "E"]).size().rename("w").reset_index()
        dir_counts = dir_counts.merge(unique_dirs, on=["A", "E"], how="left")
        w = dir_counts["w"].values

        total_samples = int(w.sum())
        target_size = total_samples / n_bins

        bin_weights = dir_counts.groupby("bin_id")["w"].sum().to_dict()
        direction_vectors = spherical_coords_to_unit_vectors(
            dir_counts["A"].values,
            dir_counts["E"].values,
            convention=convention,
        )
        c_matrix = centroids[["x", "y", "z"]].values
        dist_matrix = np.linalg.norm(
            direction_vectors[:, None, :] - c_matrix[None, :, :], axis=2
        )

        # Greedy redistribution: move directions from largest bin to nearest smaller bin
        changed = True
        iterations = 0
        while changed and iterations < 100:
            changed = False
            iterations += 1
            max_b = max(bin_weights, key=bin_weights.get)
            min_b = min(bin_weights, key=bin_weights.get)

            if bin_weights[max_b] - bin_weights[min_b] <= max(w):
                break

            idx_over = np.where(dir_counts["bin_id"].values == max_b)[0]
            idx_over = sorted(idx_over, key=lambda i: dist_matrix[i, min_b])

            for i_dir in idx_over:
                direction_weight = w[i_dir]
                # Only move if it reduces disparity
                current_diff = bin_weights[max_b] - bin_weights[min_b]
                new_diff = abs((bin_weights[max_b] - direction_weight) - (bin_weights[min_b] + direction_weight))
                if new_diff < current_diff:
                    dir_counts.at[i_dir, "bin_id"] = min_b
                    bin_weights[max_b] -= direction_weight
                    bin_weights[min_b] += direction_weight
                    changed = True
                    break

        df_out = df_out.drop(columns=["bin_id"]).merge(
            dir_counts[["A", "E", "bin_id"]], on=["A", "E"], how="left"
        )

    return df_out, centroids


def create_illumination_folds(
    df: pd.DataFrame,
    output_dir: Union[str, Path] = "splits",
) -> List[Dict]:
    """Generate and write train/test CSVs for each illumination bin fold.
    
    Args:
        df: DataFrame containing metadata and "bin_id" column.
        output_dir: Destination folder for fold CSVs and fold_summary.json.
        
    Returns:
        List of fold specification dictionaries.
    """
    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_p / "mapping.csv", index=False)

    fold_specs = []
    unique_bins = sorted(df["bin_id"].unique())

    for fold_idx, test_bin in enumerate(unique_bins):
        test_mask = df["bin_id"] == test_bin
        train_split = df.loc[~test_mask].reset_index(drop=True)
        test_split = df.loc[test_mask].reset_index(drop=True)

        train_split.to_csv(out_p / f"fold_{fold_idx}_train.csv", index=False)
        test_split.to_csv(out_p / f"fold_{fold_idx}_test.csv", index=False)

        sub_test = df[df["bin_id"] == test_bin]
        spec = {
            "fold": int(fold_idx),
            "test_bin": int(test_bin),
            "train_size": int(len(train_split)),
            "test_size": int(len(test_split)),
            "azimuth_range": [int(sub_test["A"].min()), int(sub_test["A"].max())],
            "elevation_range": [int(sub_test["E"].min()), int(sub_test["E"].max())],
        }
        fold_specs.append(spec)

    with open(out_p / "fold_summary.json", "w", encoding="utf-8") as f:
        json.dump(fold_specs, f, indent=2)

    return fold_specs
