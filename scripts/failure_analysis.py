"""Per-class failure analysis on the ensemble.

Loads the existing ensemble + tuned thresholds, runs TTA inference over the
val split the checkpoints were trained on, and saves per-class composites of
the highest-confidence false positives and false negatives.

Usage:
    uv run python scripts/failure_analysis.py
    uv run python scripts/failure_analysis.py --top-n 3 --no-tta
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader

from src.config import Config
from src.dataset import LABEL_COLS, RetinalDataset, apply_clahe, make_raw_val_transform, make_splits
from src.evaluate import TTA_AUGMENTS
from src.model import load_ensemble

_ENSEMBLE_BACKBONES = ["efficientnet_b4", "inception_resnet_v2"]
_ENSEMBLE_MEANS = [[0.485, 0.456, 0.406], [0.5, 0.5, 0.5]]
_ENSEMBLE_STDS = [[0.229, 0.224, 0.225], [0.5, 0.5, 0.5]]
_THRESHOLDS_PATH = Path("checkpoints/ensemble_thresholds.npy")
_OUT_DIR = Path("analysis/failures")
_THUMB_SIZE = 384


def _run_inference(ensemble, val_df, image_dir, transform, device, use_tta: bool):
    ds = RetinalDataset(val_df, image_dir, transform)
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    all_probs, all_labels = [], []
    with torch.no_grad():
        for left, right, labels in loader:
            left, right = left.to(device), right.to(device)
            if use_tta:
                pass_probs = []
                for aug in TTA_AUGMENTS:
                    pass_probs.append(ensemble(aug(left), aug(right)).cpu().numpy())
                probs = np.mean(pass_probs, axis=0)
            else:
                probs = ensemble(left, right).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


def _composite(records, image_dir, title: str) -> Image.Image:
    """records: list of dicts with row, prob, true. Builds a horizontal grid."""
    if not records:
        return Image.new("RGB", (_THUMB_SIZE, _THUMB_SIZE // 2), (32, 32, 32))

    cols = len(records)
    pad = 8
    header_h = 40
    cell_h = _THUMB_SIZE + 60  # space for caption under each pair
    img_w = _THUMB_SIZE * 2 + pad
    canvas = Image.new("RGB", (cols * img_w + (cols + 1) * pad, header_h + cell_h + pad), (240, 240, 240))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        cap_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except OSError:
        font = ImageFont.load_default()
        cap_font = font
    draw.text((pad, 8), title, fill=(0, 0, 0), font=font)

    for i, rec in enumerate(records):
        row = rec["row"]
        x0 = pad + i * (img_w + pad)
        y0 = header_h
        for j, side in enumerate(("Left-Fundus", "Right-Fundus")):
            img = Image.open(image_dir / row[side]).convert("RGB")
            img = apply_clahe(img)
            img.thumbnail((_THUMB_SIZE, _THUMB_SIZE))
            canvas.paste(img, (x0 + j * (_THUMB_SIZE + pad // 2), y0))
        caption = (
            f"id={int(row['ID'])}  prob={rec['prob']:.3f}  truth={int(rec['true'])}  "
            f"all_true={','.join(c for c in LABEL_COLS if int(row[c]) == 1) or 'none'}"
        )
        draw.text((x0, y0 + _THUMB_SIZE + 6), caption, fill=(0, 0, 0), font=cap_font)
    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-n", type=int, default=4, help="Top N FP and Top N FN per class")
    parser.add_argument("--no-tta", action="store_true")
    args = parser.parse_args()
    use_tta = not args.no_tta

    config = Config()
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    checkpoints = [Path("checkpoints") / b / "best_model.pt" for b in _ENSEMBLE_BACKBONES]
    missing = [str(p) for p in checkpoints if not p.exists()]
    if missing:
        raise SystemExit("Missing checkpoints:\n  " + "\n  ".join(missing))
    if not _THRESHOLDS_PATH.exists():
        raise SystemExit(f"Missing {_THRESHOLDS_PATH} - run main.py option 3 first.")

    print(f"Device: {device} | TTA: {use_tta}")
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

    _, val_df = make_splits(config)
    val_df = val_df.reset_index(drop=True)
    transform = make_raw_val_transform(config.image_size)
    print(f"Inference on {len(val_df)} val records...")
    probs, labels = _run_inference(ensemble, val_df, config.image_dir, transform, device, use_tta)

    thresholds = np.load(_THRESHOLDS_PATH)
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_lines = ["# Failure Analysis", ""]
    for i, label in enumerate(config.labels):
        thr = thresholds[i]
        cls_probs = probs[:, i]
        cls_truth = labels[:, i].astype(int)
        cls_pred = (cls_probs >= thr).astype(int)

        fp_idx = np.where((cls_pred == 1) & (cls_truth == 0))[0]
        fn_idx = np.where((cls_pred == 0) & (cls_truth == 1))[0]

        # Worst FP: highest prob among false positives. Worst FN: lowest prob among false negatives.
        fp_idx = fp_idx[np.argsort(-cls_probs[fp_idx])][: args.top_n]
        fn_idx = fn_idx[np.argsort(cls_probs[fn_idx])][: args.top_n]

        fp_records = [{"row": val_df.iloc[idx], "prob": float(cls_probs[idx]), "true": cls_truth[idx]} for idx in fp_idx]
        fn_records = [{"row": val_df.iloc[idx], "prob": float(cls_probs[idx]), "true": cls_truth[idx]} for idx in fn_idx]

        fp_path = _OUT_DIR / f"{label}_fp.png"
        fn_path = _OUT_DIR / f"{label}_fn.png"
        _composite(fp_records, config.image_dir, f"{label}: top {len(fp_records)} false positives  (thr={thr:.3f})").save(fp_path)
        _composite(fn_records, config.image_dir, f"{label}: top {len(fn_records)} false negatives  (thr={thr:.3f})").save(fn_path)

        n_pos = int(cls_truth.sum())
        n_pred_pos = int(cls_pred.sum())
        n_fp_total = int(((cls_pred == 1) & (cls_truth == 0)).sum())
        n_fn_total = int(((cls_pred == 0) & (cls_truth == 1)).sum())
        print(f"  {label}: thr={thr:.3f}  pos={n_pos}  pred_pos={n_pred_pos}  "
              f"FP={len(fp_records)}/{n_fp_total}  FN={len(fn_records)}/{n_fn_total}")

        summary_lines.append(f"## {label}")
        summary_lines.append(f"- threshold: {thr:.3f}")
        summary_lines.append(f"- positives in val: {n_pos}")
        summary_lines.append(f"- top FP: ![]({fp_path.as_posix()})  *(hypothesis: TODO)*")
        summary_lines.append(f"- top FN: ![]({fn_path.as_posix()})  *(hypothesis: TODO)*")
        summary_lines.append("")

    summary_path = _OUT_DIR / "summary.md"
    summary_path.write_text("\n".join(summary_lines))
    print(f"\nSaved composites to {_OUT_DIR}/")
    print(f"Summary: {summary_path}")
    print("Next: open the composites, replace each 'hypothesis: TODO' in summary.md with one line.")


if __name__ == "__main__":
    main()
