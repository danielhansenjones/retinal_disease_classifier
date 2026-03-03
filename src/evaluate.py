import numpy as np
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


def tune_thresholds(y_true: np.ndarray, y_prob: np.ndarray, labels: list[str]) -> np.ndarray:
    """Per-class threshold that maximises F1 on the provided split."""
    thresholds = np.full(len(labels), 0.5)
    for i in range(len(labels)):
        precision, recall, thresh = precision_recall_curve(y_true[:, i], y_prob[:, i])
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        if len(thresh) > 0:
            thresholds[i] = thresh[f1[:-1].argmax()]
    return thresholds