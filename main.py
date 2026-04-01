from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import Config
from src.dataset import RetinalDataset, make_raw_val_transform, make_splits
from src.evaluate import TTA_AUGMENTS, compute_metrics, tune_thresholds
from src.model import load_ensemble
from src.train import train

_CONFIGS = {
    "1": dict(backbone="efficientnet_b4"),
    "2": dict(backbone="inception_resnet_v2", batch_size_finetune=16),
}

_ENSEMBLE_BACKBONES = ["efficientnet_b4", "inception_resnet_v2"]
_ENSEMBLE_MEANS = [[0.485, 0.456, 0.406], [0.5, 0.5, 0.5]]
_ENSEMBLE_STDS = [[0.229, 0.224, 0.225], [0.5, 0.5, 0.5]]


def run_ensemble():
    config = Config()
    checkpoints = [Path("checkpoints") / b / "best_model.pt" for b in _ENSEMBLE_BACKBONES]
    missing = [str(p) for p in checkpoints if not p.exists()]
    if missing:
        raise SystemExit(f"Missing checkpoints - train both models first:\n  " + "\n  ".join(missing))

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    _, val_df = make_splits(config)
    val_ds = RetinalDataset(val_df, config.image_dir, make_raw_val_transform(config.image_size))
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False,
                            num_workers=config.num_workers, pin_memory=True)

    print("Loading ensemble...")
    ensemble = load_ensemble(
        checkpoints=checkpoints,
        backbone_names=_ENSEMBLE_BACKBONES,
        norm_means=_ENSEMBLE_MEANS,
        norm_stds=_ENSEMBLE_STDS,
        num_classes=len(config.labels),
        dropout=config.dropout,
        device=device,
    )

    print("Running inference with TTA (4 views x 2 models)...")
    all_probs, all_labels = [], []
    with torch.no_grad():
        for left, right, labels in val_loader:
            left, right = left.to(device), right.to(device)
            pass_probs = []
            for aug in TTA_AUGMENTS:
                # EnsembleModel already returns sigmoid-averaged probs - no sigmoid here
                pass_probs.append(ensemble(aug(left), aug(right)).cpu().numpy())
            all_probs.append(np.mean(pass_probs, axis=0))
            all_labels.append(labels.numpy())

    y_prob = np.concatenate(all_probs)
    y_true = np.concatenate(all_labels)

    thresholds = tune_thresholds(y_true, y_prob, config.labels)
    np.save(Path("checkpoints/ensemble_thresholds.npy"), thresholds)
    metrics = compute_metrics(y_true, y_prob, thresholds, config.labels)

    print("\n--- Ensemble results (tuned thresholds) ---")
    for label in config.labels:
        m = metrics[label]
        print(f"  {label}: auc={m['auc']:.3f}  f1={m['f1']:.3f}  "
              f"prec={m['precision']:.3f}  rec={m['recall']:.3f}  thresh={thresholds[config.labels.index(label)]:.3f}")
    print(f"  macro_auc={metrics['macro_auc']:.4f}")


if __name__ == "__main__":
    while True:
        print("\nSelect an option:")
        print("  1) Train EfficientNet-B4")
        print("  2) Train Inception-ResNet-v2")
        print("  3) Evaluate ensemble (both checkpoints must exist)")
        print("  4) Exit")
        choice = input("Enter 1, 2, 3, or 4: ").strip()
        if choice in _CONFIGS:
            train(Config(**_CONFIGS[choice]))
        elif choice == "3":
            run_ensemble()
        elif choice == "4":
            break
        else:
            print(f"Invalid choice '{choice}' - enter 1, 2, 3, or 4.")