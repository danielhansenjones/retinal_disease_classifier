import numpy as np

from src.evaluate import compute_metrics, tune_thresholds


def test_thresholds_in_unit_interval_and_match_label_count():
    rng = np.random.default_rng(0)
    n, k = 200, 8
    y_true = rng.integers(0, 2, size=(n, k))
    y_prob = rng.random((n, k))
    labels = [f"L{i}" for i in range(k)]

    thresholds = tune_thresholds(y_true, y_prob, labels)
    assert thresholds.shape == (k,)
    assert np.all(thresholds >= 0.0)
    assert np.all(thresholds <= 1.0)


def test_compute_metrics_returns_macro_auc_and_per_label_keys():
    rng = np.random.default_rng(1)
    n, k = 100, 3
    y_true = rng.integers(0, 2, size=(n, k))
    y_prob = rng.random((n, k))
    labels = ["A", "B", "C"]
    thresholds = np.full(k, 0.5)

    metrics = compute_metrics(y_true, y_prob, thresholds, labels)
    assert "macro_auc" in metrics
    for label in labels:
        assert {"auc", "f1", "precision", "recall"} <= metrics[label].keys()
        assert 0.0 <= metrics[label]["f1"] <= 1.0
        assert 0.0 <= metrics[label]["precision"] <= 1.0
        assert 0.0 <= metrics[label]["recall"] <= 1.0