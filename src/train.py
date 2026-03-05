import json

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from src.config import Config
from src.dataset import RetinalDataset, LABEL_COLS, make_splits, TRAIN_TRANSFORMS, VAL_TRANSFORMS
from src.evaluate import compute_metrics, tune_thresholds
from src.model import build_model, freeze_backbone, unfreeze_backbone


def compute_pos_weight(train_df, device):
    labels = train_df[LABEL_COLS].values.astype(np.float32)
    pos = labels.sum(axis=0)
    neg = len(labels) - pos
    pos_weight = torch.tensor(neg / np.maximum(pos, 1), dtype=torch.float32).to(device)
    return pos_weight


def run_epoch(model, loader, criterion, optimizer, scaler, device, train: bool):
    model.train(train)
    total_loss = 0.0
    all_probs, all_labels = [], []

    with torch.set_grad_enabled(train):
        for left, right, labels in loader:
            left, right, labels = left.to(device), right.to(device), labels.to(device)

            with autocast("cuda"):
                logits = model(left, right)
                loss = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item() * len(labels)
            all_probs.append(torch.sigmoid(logits).detach().cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    y_prob = np.concatenate(all_probs)
    y_true = np.concatenate(all_labels)
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, y_true, y_prob


def make_loader(ds, batch_size, shuffle, num_workers):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True,
                      persistent_workers=True, prefetch_factor=2)


def train(config: Config):
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_df, val_df = make_splits(config)
    train_ds = RetinalDataset(train_df, config.image_dir, TRAIN_TRANSFORMS)
    val_ds = RetinalDataset(val_df, config.image_dir, VAL_TRANSFORMS)

    model = build_model(config.backbone, len(config.labels), config.dropout).to(device)
    pos_weight = compute_pos_weight(train_df, device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scaler = GradScaler("cuda")

    best_auc = 0.0
    patience_counter = 0

    with mlflow.start_run():
        mlflow.log_params({
            "backbone": config.backbone,
            "image_size": config.image_size,
            "dropout": config.dropout,
            "batch_size": config.batch_size,
            "batch_size_finetune": config.batch_size_finetune,
            "lr_head": config.lr_head,
            "lr_finetune": config.lr_finetune,
            "epochs_frozen": config.epochs_frozen,
            "epochs_unfrozen": config.epochs_unfrozen,
            "patience": config.patience,
            "val_split": config.val_split,
        })

        # Phase 1 - frozen backbone, train head only
        freeze_backbone(model)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr_head)
        train_loader = make_loader(train_ds, config.batch_size, shuffle=True, num_workers=config.num_workers)
        val_loader = make_loader(val_ds, config.batch_size, shuffle=False, num_workers=config.num_workers)

        print("\n--- Phase 1: frozen backbone ---")
        for epoch in range(config.epochs_frozen):
            train_loss, _, _ = run_epoch(model, train_loader, criterion, optimizer, scaler, device, train=True)
            val_loss, y_true, y_prob = run_epoch(model, val_loader, criterion, optimizer, scaler, device, train=False)
            thresholds = np.full(len(config.labels), 0.5)
            metrics = compute_metrics(y_true, y_prob, thresholds, config.labels)
            macro_auc = metrics["macro_auc"]
            mlflow.log_metrics({
                "p1/train_loss": train_loss,
                "p1/val_loss": val_loss,
                "p1/val_auc": macro_auc,
            }, step=epoch)
            print(f"Epoch {epoch+1}/{config.epochs_frozen} | train_loss={train_loss:.4f} "
                  f"val_loss={val_loss:.4f} val_auc={macro_auc:.4f}")

        # Phase 2 - unfreeze all layers, smaller batch to fit full gradients in VRAM
        unfreeze_backbone(model)
        model.enable_grad_checkpointing()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr_finetune)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs_unfrozen)
        train_loader = make_loader(train_ds, config.batch_size_finetune, shuffle=True, num_workers=config.num_workers)
        val_loader = make_loader(val_ds, config.batch_size_finetune, shuffle=False, num_workers=config.num_workers)

        start_epoch = 0
        ckpt_path = config.checkpoint_dir / "best_model.pt"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
            if "optimizer" in ckpt:
                model.load_state_dict(ckpt["model"])
                optimizer.load_state_dict(ckpt["optimizer"])
                scheduler.load_state_dict(ckpt["scheduler"])
                scaler.load_state_dict(ckpt["scaler"])
                start_epoch = ckpt["epoch"] + 1
                best_auc = ckpt["best_auc"]
                patience_counter = ckpt["patience_counter"]
                print(f"Resumed from epoch {start_epoch}, best_auc={best_auc:.4f}")

        print("\n--- Phase 2: full fine-tune ---")
        for epoch in range(start_epoch, config.epochs_unfrozen):
            train_loss, _, _ = run_epoch(model, train_loader, criterion, optimizer, scaler, device, train=True)
            val_loss, y_true, y_prob = run_epoch(model, val_loader, criterion, optimizer, scaler, device, train=False)
            thresholds = np.full(len(config.labels), 0.5)
            metrics = compute_metrics(y_true, y_prob, thresholds, config.labels)
            macro_auc = metrics["macro_auc"]
            scheduler.step()

            epoch_metrics = {
                "p2/train_loss": train_loss,
                "p2/val_loss": val_loss,
                "p2/val_auc": macro_auc,
            }
            for label in config.labels:
                epoch_metrics[f"p2/auc_{label}"] = metrics[label]["auc"]
            mlflow.log_metrics(epoch_metrics, step=epoch)

            print(f"Epoch {epoch+1}/{config.epochs_unfrozen} | train_loss={train_loss:.4f} "
                  f"val_loss={val_loss:.4f} val_auc={macro_auc:.4f}")

            if macro_auc > best_auc:
                best_auc = macro_auc
                patience_counter = 0
                config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "best_auc": best_auc,
                    "patience_counter": patience_counter,
                }, config.checkpoint_dir / "best_model.pt")
                print(f"  -> saved best model (auc={best_auc:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= config.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        # Tune thresholds on val set using best model
        ckpt = torch.load(config.checkpoint_dir / "best_model.pt", weights_only=False, map_location=device)
        model.load_state_dict(ckpt["model"])
        _, y_true, y_prob = run_epoch(model, val_loader, criterion, optimizer, scaler, device, train=False)
        thresholds = tune_thresholds(y_true, y_prob, config.labels)
        final_metrics = compute_metrics(y_true, y_prob, thresholds, config.labels)

        print("\n--- Final per-class metrics (tuned thresholds) ---")
        final_mlflow_metrics = {"final/macro_auc": final_metrics["macro_auc"]}
        for label in config.labels:
            m = final_metrics[label]
            print(f"  {label}: auc={m['auc']:.3f} f1={m['f1']:.3f} "
                  f"prec={m['precision']:.3f} rec={m['recall']:.3f} thresh={thresholds[config.labels.index(label)]:.3f}")
            final_mlflow_metrics[f"final/auc_{label}"] = m["auc"]
            final_mlflow_metrics[f"final/f1_{label}"] = m["f1"]
            final_mlflow_metrics[f"final/precision_{label}"] = m["precision"]
            final_mlflow_metrics[f"final/recall_{label}"] = m["recall"]
        print(f"  macro_auc={final_metrics['macro_auc']:.4f}")
        mlflow.log_metrics(final_mlflow_metrics)

        np.save(config.checkpoint_dir / "thresholds.npy", thresholds)
        with open(config.checkpoint_dir / "metrics.json", "w") as f:
            json.dump(final_metrics, f, indent=2)

        mlflow.log_artifact(str(config.checkpoint_dir / "thresholds.npy"))
        mlflow.log_artifact(str(config.checkpoint_dir / "metrics.json"))
        mlflow.pytorch.log_model(
            model,
            name="model",
            registered_model_name="retinal-disease-classifier",
            pip_requirements=["torch", "torchvision", "timm", "numpy"],
        )

    return model, thresholds, final_metrics