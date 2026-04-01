from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, Query, UploadFile
from PIL import Image
from pydantic import BaseModel

from src.config import Config
from src.dataset import apply_clahe, make_raw_val_transform
from src.evaluate import TTA_AUGMENTS
from src.model import load_ensemble

_ENSEMBLE_BACKBONES = ["efficientnet_b4", "inception_resnet_v2"]
_ENSEMBLE_MEANS = [[0.485, 0.456, 0.406], [0.5, 0.5, 0.5]]
_ENSEMBLE_STDS = [[0.229, 0.224, 0.225], [0.5, 0.5, 0.5]]
_THRESHOLDS_PATH = Path("checkpoints/ensemble_thresholds.npy")


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = Config()
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    checkpoints = [Path("checkpoints") / b / "best_model.pt" for b in _ENSEMBLE_BACKBONES]
    missing = [str(p) for p in checkpoints if not p.exists()]
    if missing:
        raise RuntimeError("Missing checkpoints:\n  " + "\n  ".join(missing))

    if not _THRESHOLDS_PATH.exists():
        raise RuntimeError(
            f"Missing {_THRESHOLDS_PATH} - run ensemble evaluation first (option 3 in main.py)"
        )

    app.state.ensemble = load_ensemble(
        checkpoints=checkpoints,
        backbone_names=_ENSEMBLE_BACKBONES,
        norm_means=_ENSEMBLE_MEANS,
        norm_stds=_ENSEMBLE_STDS,
        num_classes=len(config.labels),
        dropout=config.dropout,
        device=device,
    )
    app.state.transform = make_raw_val_transform(config.image_size)
    app.state.thresholds = np.load(_THRESHOLDS_PATH)
    app.state.labels = config.labels
    app.state.device = device

    yield


app = FastAPI(title="Retinal Disease Classifier", lifespan=lifespan)


class Prediction(BaseModel):
    probabilities: dict[str, float]
    predictions: dict[str, bool]


def _preprocess(raw_bytes: bytes, transform) -> torch.Tensor:
    img = Image.open(BytesIO(raw_bytes)).convert("RGB")
    img = apply_clahe(img)
    return transform(img).unsqueeze(0)


@app.post("/predict", response_model=Prediction)
async def predict(
    left: UploadFile,
    right: UploadFile,
    tta: bool = Query(True, description="Enable test-time augmentation (4 views)"),
):
    left_tensor = _preprocess(await left.read(), app.state.transform).to(app.state.device)
    right_tensor = _preprocess(await right.read(), app.state.transform).to(app.state.device)

    with torch.no_grad():
        if tta:
            pass_probs = []
            for aug in TTA_AUGMENTS:
                pass_probs.append(
                    app.state.ensemble(aug(left_tensor), aug(right_tensor)).cpu().numpy()
                )
            probs = np.mean(pass_probs, axis=0)[0]
        else:
            probs = app.state.ensemble(left_tensor, right_tensor).cpu().numpy()[0]

    labels = app.state.labels
    thresholds = app.state.thresholds
    return Prediction(
        probabilities={label: round(float(probs[i]), 4) for i, label in enumerate(labels)},
        predictions={label: bool(probs[i] >= thresholds[i]) for i, label in enumerate(labels)},
    )