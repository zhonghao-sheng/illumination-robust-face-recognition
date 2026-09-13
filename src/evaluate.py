"""Evaluation suite for multi-group illumination robustness and error analysis."""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate standard multi-class classification metrics.
    
    Args:
        y_true: Ground truth labels (N,).
        y_pred: Predicted labels (N,).
        
    Returns:
        Dictionary containing overall accuracy, macro precision, macro recall, and macro F1.
    """
    acc = float(accuracy_score(y_true, y_pred))
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "accuracy": acc,
        "macro_precision": float(prec),
        "macro_recall": float(rec),
        "macro_f1": float(f1),
    }


def evaluate_by_group(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group_ids: np.ndarray,
) -> Dict[int, Dict[str, Any]]:
    """Calculate performance metrics segmented across illumination groups/bins.
    
    Args:
        y_true: Ground truth labels (N,).
        y_pred: Predicted labels (N,).
        group_ids: Illumination bin identifiers for each sample (N,).
        
    Returns:
        Dictionary mapping group ID to sample count, correct count, accuracy, and error rate.
    """
    results = {}
    unique_groups = sorted(list(set(group_ids)))

    for g in unique_groups:
        mask = group_ids == g
        g_true = y_true[mask]
        g_pred = y_pred[mask]
        n_samples = int(len(g_true))

        if n_samples == 0:
            continue

        n_correct = int(np.sum(g_true == g_pred))
        acc = n_correct / n_samples
        results[int(g)] = {
            "n_samples": n_samples,
            "n_correct": n_correct,
            "accuracy": float(acc),
            "error_rate": float(1.0 - acc),
        }

    return results


def compute_worst_group_metrics(group_results: Dict[int, Dict[str, Any]]) -> Dict[str, float]:
    """Compute subpopulation robustness and disparity metrics across illumination groups.
    
    Args:
        group_results: Output from `evaluate_by_group`.
        
    Returns:
        Dictionary containing mean accuracy, worst-group accuracy, best-group accuracy,
        and group disparity (best - worst).
    """
    accuracies = [data["accuracy"] for data in group_results.values()]
    if not accuracies:
        return {"mean_group_acc": 0.0, "worst_group_acc": 0.0, "best_group_acc": 0.0, "disparity": 0.0}

    mean_acc = float(np.mean(accuracies))
    worst_acc = float(np.min(accuracies))
    best_acc = float(np.max(accuracies))
    disparity = best_acc - worst_acc

    return {
        "mean_group_acc": mean_acc,
        "worst_group_acc": worst_acc,
        "best_group_acc": best_acc,
        "disparity": disparity,
        "std_group_acc": float(np.std(accuracies)),
    }


def generate_confusion_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """Identify top pairs of confused classes to support error analysis.
    
    Args:
        y_true: True class indices or names.
        y_pred: Predicted class indices or names.
        class_names: Optional list of class label strings.
        top_k: Number of highest-frequency confusion pairs to return.
        
    Returns:
        List of dictionaries with true_class, pred_class, count, and error frequency.
    """
    labels = sorted(list(set(y_true).union(set(y_pred))))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    np.fill_diagonal(cm, 0)  # zero out correct classifications

    pairs = []
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j] > 0:
                c_true = class_names[i] if class_names else str(labels[i])
                c_pred = class_names[j] if class_names else str(labels[j])
                pairs.append({
                    "true_class": c_true,
                    "pred_class": c_pred,
                    "confusion_count": int(cm[i, j]),
                })

    pairs = sorted(pairs, key=lambda p: p["confusion_count"], reverse=True)
    return pairs[:top_k]
