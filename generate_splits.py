"""CLI script to deterministically recreate spherical illumination splits."""

import argparse
import sys
from pathlib import Path
from src.splits import parse_dataset_metadata, generate_spherical_bins, create_illumination_folds


def main():
    parser = argparse.ArgumentParser(
        description="Generate 10-fold spherical illumination splits from Extended Yale B dataset."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/YaleFace/cropped",
        help="Path to cropped Yale B face image folder.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="splits",
        help="Directory to save generated CSV splits and summary JSON.",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=10,
        help="Number of spherical illumination clusters/folds.",
    )
    parser.add_argument(
        "--convention",
        type=str,
        default="physical",
        choices=["physical", "legacy"],
        help="Spherical coordinate convention: physical or legacy alternate mapping.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic k-means clustering.",
    )
    args = parser.parse_args()

    data_p = Path(args.data_dir)
    if not data_p.exists():
        print(f"Error: Dataset directory does not exist: {data_p}", file=sys.stderr)
        print("Please place the Extended Yale B dataset under data/YaleFace/cropped/ or pass --data-dir.", file=sys.stderr)
        return 1

    print(f"Scanning dataset in {data_p}...")
    df = parse_dataset_metadata(data_p)
    n_subjects = df["person"].nunique()
    print(f"Found {len(df)} images across {n_subjects} subjects.")

    print(f"Clustering illumination directions into {args.n_bins} spherical bins (convention: {args.convention})...")
    df_bins, centroids = generate_spherical_bins(
        df, n_bins=args.n_bins, seed=args.seed, convention=args.convention
    )

    print(f"Writing splits to {args.output_dir}...")
    specs = create_illumination_folds(df_bins, output_dir=args.output_dir)

    print("\nSpherical Split Generation Complete:")
    for s in specs:
        fold_id = s["fold"]
        test_bin = s["test_bin"]
        a_range = s["azimuth_range"]
        e_range = s["elevation_range"]
        n_tr = s["train_size"]
        n_te = s["test_size"]
        print(f"  Fold {fold_id}: Test Bin {test_bin} (Azimuth: {a_range}, Elevation: {e_range}) | Train={n_tr}, Test={n_te}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
