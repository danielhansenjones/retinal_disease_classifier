import numpy as np
import torch
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, precision_recall_curve


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, thresholds: np.ndarray, labels: list[str]) -> dict:
    """Per-class AUC, F1, precision, recall at given thresholds."""
    metrics = {}
    for i, label in enumerate(labels):
        y_pred = (y_prob[:, i] >= thresholds[i]).astype(int)
        metrics[label] = {
            "auc": roc_auc_score(y_true[:, i], y_prob[:, i]),
            "f1": f1_score(y_true[:, i], y_pred, zero_division=0),
            "precision": precision_score(y_true[:, i], y_pred, zero_division=0),
            "recall": recall_score(y_true[:, i], y_pred, zero_division=0),
        }
    auc_vals = [metrics[l]["auc"] for l in labels]
    metrics["macro_auc"] = float(np.mean(auc_vals))
    return metrics


def run_tta(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_probs, all_labels = [], []

    # Four deterministic views - all valid for fundus images (no natural orientation)
    augments = [
        lambda t: t,
        lambda t: torch.flip(t, dims=[-1]),
        lambda t: torch.flip(t, dims=[-2]),
        lambda t: torch.rot90(t, k=1, dims=[-2, -1]),
    ]

    with torch.no_grad():
        for left, right, labels in loader:
            left, right = left.to(device), right.to(device)
            pass_probs = []
            for aug in augments:
                logits = model(aug(left), aug(right))
                pass_probs.append(torch.sigmoid(logits).cpu().numpy())
            all_probs.append(np.mean(pass_probs, axis=0))
            all_labels.append(labels.numpy())

    return np.concatenate(all_labels), np.concatenate(all_probs)


def tune_thresholds(y_true: np.ndarray, y_prob: np.ndarray, labels: list[str]) -> np.ndarray:
    """Per-class threshold that maximises F1 on the provided split."""
    thresholds = np.full(len(labels), 0.5)
    for i in range(len(labels)):
        precision, recall, thresh = precision_recall_curve(y_true[:, i], y_prob[:, i])
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        if len(thresh) > 0:
            thresholds[i] = thresh[f1[:-1].argmax()]
    return thresholds