"""Grouped nested evaluation of LDA+SVM, a custom CNN, and EfficientNet-B0."""

import argparse
import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold

from src.data_loader import FaceDataset, encode_labels, load_classical_features, verify_image_paths
from src.evaluate import evaluate_predictions
from src.models import IlluminationCNN, LdaSvmClassifier, build_efficientnet_b0
from src.train import predict_model, seed_everything, train_fixed_epochs


SVM_C = (0.001, 0.01, 0.1)
CNN_KERNELS = (3, 5, 7)
EFFNET_LR = (1e-3, 5e-4, 3e-4)
EFFNET_WEIGHT_DECAY = (1e-5, 1e-4, 5e-4)


def inner_partitions(frame: pd.DataFrame, n_splits: int = 3):
    groups = frame["bin_id"].to_numpy()
    if len(np.unique(groups)) < n_splits:
        raise ValueError(f"Need at least {n_splits} training lighting bins for grouped inner CV")
    yield from GroupKFold(n_splits=n_splits).split(frame, groups=groups)


def choose_svm(frame, labels, features):
    scores = {}
    for c in SVM_C:
        fold_scores = []
        for train_indices, validation_indices in inner_partitions(frame):
            model = LdaSvmClassifier(C=c).fit(features[train_indices], labels[train_indices])
            predictions = model.predict(features[validation_indices])
            fold_scores.append(evaluate_predictions(labels[validation_indices], predictions)["macro_f1"])
        scores[c] = float(np.mean(fold_scores))
    return max(scores, key=scores.get), scores


def build_model(name: str, n_classes: int, config: dict, pretrained: bool):
    if name == "cnn":
        return IlluminationCNN(n_classes=n_classes, kernel_size=config["kernel_size"])
    if name == "efficientnet":
        return build_efficientnet_b0(n_classes=n_classes, pretrained=pretrained)
    raise ValueError(f"Unknown neural model: {name}")


def choose_neural(name, frame, labels, data_root, args, device, n_classes):
    if name == "cnn":
        configs = [{"kernel_size": value, "learning_rate": args.cnn_lr, "weight_decay": args.cnn_weight_decay}
                   for value in CNN_KERNELS]
        epochs = args.cnn_inner_epochs
    else:
        configs = [{"learning_rate": lr, "weight_decay": wd}
                   for lr, wd in itertools.product(EFFNET_LR, EFFNET_WEIGHT_DECAY)]
        epochs = args.effnet_inner_epochs
    scores = {}
    for config_index, config in enumerate(configs):
        fold_scores = []
        for inner_index, (train_indices, validation_indices) in enumerate(inner_partitions(frame)):
            seed = args.seed + inner_index
            seed_everything(seed)
            model = build_model(name, n_classes, config, args.pretrained)
            training = FaceDataset(
                frame.iloc[train_indices], labels[train_indices], data_root,
                name, args.cnn_height, args.cnn_width, training=True, seed=seed,
            )
            validation = FaceDataset(
                frame.iloc[validation_indices], labels[validation_indices], data_root,
                name, args.cnn_height, args.cnn_width,
            )
            model = train_fixed_epochs(
                model, training, epochs=epochs, batch_size=args.batch_size,
                learning_rate=config["learning_rate"], weight_decay=config["weight_decay"],
                device=device, seed=seed,
            )
            predictions = predict_model(model, validation, batch_size=args.batch_size, device=device)
            fold_scores.append(evaluate_predictions(labels[validation_indices], predictions)["macro_f1"])
        scores[config_index] = float(np.mean(fold_scores))
    selected_index = max(scores, key=scores.get)
    return configs[selected_index], [
        {"config": config, "mean_inner_macro_f1": scores[index]}
        for index, config in enumerate(configs)
    ]


def load_fold(splits_dir: Path, fold: int):
    train_path = splits_dir / f"fold_{fold}_train.csv"
    test_path = splits_dir / f"fold_{fold}_test.csv"
    if not train_path.is_file() or not test_path.is_file():
        raise FileNotFoundError(f"Missing split CSVs for fold {fold}; run generate_splits.py first")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    required = {"path", "person", "bin_id"}
    if not required.issubset(train.columns) or not required.issubset(test.columns):
        raise ValueError("Split CSVs require path, person and bin_id columns")
    if set(train["bin_id"]) & set(test["bin_id"]):
        raise ValueError("Training and test illumination bins overlap")
    if set(train["path"]) & set(test["path"]):
        raise ValueError("Training and test image paths overlap")
    if len(test["bin_id"].unique()) != 1:
        raise ValueError("Each outer test fold must contain one lighting bin")
    if not train["person"].notna().all() or not test["person"].notna().all():
        raise ValueError("Missing identity label")
    return train.reset_index(drop=True), test.reset_index(drop=True)


def evaluate_fold(fold: int, args, device: torch.device):
    train, test = load_fold(Path(args.splits_dir), fold)
    verify_image_paths(train, args.data_root)
    verify_image_paths(test, args.data_root)
    (train_labels,), classes = encode_labels(train["person"].tolist())
    class_to_index = {label: index for index, label in enumerate(classes)}
    if set(test["person"]) - set(class_to_index):
        raise ValueError("Outer test contains identities absent from training")
    # The test labels are encoded only after all models are fitted.
    fitted = {}
    chosen = {}

    if "svm" in args.models:
        features = load_classical_features(train, args.data_root, args.svm_height, args.svm_width)
        c, scores = choose_svm(train, train_labels, features)
        fitted["svm"] = LdaSvmClassifier(C=c).fit(features, train_labels)
        chosen["svm"] = {"C": c, "inner_macro_f1": scores}

    for name in ("cnn", "efficientnet"):
        if name not in args.models:
            continue
        config, scores = choose_neural(name, train, train_labels, args.data_root, args, device, len(classes))
        seed_everything(args.seed + fold * 100)
        model = build_model(name, len(classes), config, args.pretrained)
        dataset = FaceDataset(
            train, train_labels, args.data_root, name, args.cnn_height, args.cnn_width,
            training=True, seed=args.seed + fold * 100,
        )
        epochs = args.cnn_outer_epochs if name == "cnn" else args.effnet_outer_epochs
        fitted[name] = train_fixed_epochs(
            model, dataset, epochs=epochs, batch_size=args.batch_size,
            learning_rate=config["learning_rate"], weight_decay=config["weight_decay"],
            device=device, seed=args.seed + fold * 100,
        )
        chosen[name] = {"selected": config, "inner_search": scores}

    test_labels = np.asarray([class_to_index[str(value)] for value in test["person"]], dtype=np.int64)
    metrics = {}
    for name, model in fitted.items():
        if name == "svm":
            test_features = load_classical_features(test, args.data_root, args.svm_height, args.svm_width)
            predictions = model.predict(test_features)
        else:
            dataset = FaceDataset(
                test, test_labels, args.data_root, name, args.cnn_height, args.cnn_width,
            )
            predictions = predict_model(model, dataset, batch_size=args.batch_size, device=device)
        metrics[name] = evaluate_predictions(test_labels, predictions)
    return {
        "fold": fold,
        "test_bin": int(test["bin_id"].iloc[0]),
        "train_size": len(train),
        "test_size": len(test),
        "selected": chosen,
        "metrics": metrics,
    }


def summarize(fold_results, models):
    summary = {}
    for model in models:
        per_fold = [result["metrics"][model] for result in fold_results]
        summary[model] = {
            key: {
                "mean": float(np.mean([item[key] for item in per_fold])),
                "sd": float(np.std([item[key] for item in per_fold], ddof=1)) if len(per_fold) > 1 else None,
            }
            for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1")
        }
        choices = [result["selected"][model] for result in fold_results]
        if model == "svm":
            summary[model]["selected_C_counts"] = dict(Counter(str(item["C"]) for item in choices))
        elif model == "cnn":
            summary[model]["selected_kernel_counts"] = dict(
                Counter(str(item["selected"]["kernel_size"]) for item in choices)
            )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/YaleFace/cropped")
    parser.add_argument("--splits-dir", default="splits")
    parser.add_argument("--folds", default="all", help="all or comma-separated outer fold numbers")
    parser.add_argument("--models", default="svm,cnn,efficientnet")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--svm-height", type=int, default=48)
    parser.add_argument("--svm-width", type=int, default=40)
    parser.add_argument("--cnn-height", type=int, default=96)
    parser.add_argument("--cnn-width", type=int, default=84)
    parser.add_argument("--cnn-lr", type=float, default=1e-3)
    parser.add_argument("--cnn-weight-decay", type=float, default=1e-4)
    parser.add_argument("--cnn-inner-epochs", type=int, default=10)
    parser.add_argument("--cnn-outer-epochs", type=int, default=18)
    parser.add_argument("--effnet-inner-epochs", type=int, default=3)
    parser.add_argument("--effnet-outer-epochs", type=int, default=15)
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false",
                        help="Skip ImageNet weights for an offline smoke run")
    parser.add_argument("--output", help="Optional JSON result file (keep local until reviewed)")
    args = parser.parse_args()
    args.models = tuple(part.strip() for part in args.models.split(",") if part.strip())
    if not args.models or set(args.models) - {"svm", "cnn", "efficientnet"}:
        parser.error("--models must contain svm, cnn and/or efficientnet")
    folds = list(range(10)) if args.folds == "all" else [int(value) for value in args.folds.split(",")]
    if not folds or any(fold < 0 or fold > 9 for fold in folds):
        parser.error("--folds must be all or fold numbers 0 through 9")
    if args.device == "auto":
        name = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        name = args.device
    device = torch.device(name)
    results = []
    for fold in folds:
        result = evaluate_fold(fold, args, device)
        results.append(result)
        print(f"Fold {fold} (test bin {result['test_bin']}, n={result['test_size']}):")
        for model, metrics in result["metrics"].items():
            print(f"  {model}: accuracy={metrics['accuracy']:.4f}, macro F1={metrics['macro_f1']:.4f}")
    document = {
        "protocol": "10 outer lighting-bin folds; 3 grouped inner folds; macro-F1 selection",
        "seed": args.seed,
        "models": args.models,
        "folds": results,
        "summary": summarize(results, args.models),
    }
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        print(f"Saved run metrics to {output}")
    print(json.dumps(document["summary"], indent=2))


if __name__ == "__main__":
    main()
