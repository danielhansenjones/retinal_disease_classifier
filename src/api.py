import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Query, Request, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from src.config import Config
from src.dataset import apply_clahe, make_raw_val_transform
from src.evaluate import TTA_AUGMENTS
from src.model import load_ensemble

_ENSEMBLE_BACKBONES = ["efficientnet_b4", "inception_resnet_v2"]
_ENSEMBLE_MEANS = [[0.485, 0.456, 0.406], [0.5, 0.5, 0.5]]
_ENSEMBLE_STDS = [[0.229, 0.224, 0.225], [0.5, 0.5, 0.5]]
_THRESHOLDS_PATH = Path("checkpoints/ensemble_thresholds.npy")

_MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB per upload
_MAX_PIXELS = 50_000_000  # blocks decompression bombs
_ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/tiff", "image/bmp"}
_INFERENCE_CONCURRENCY = 1  # single GPU; serialize requests so a flood does not OOM

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("api")


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
    app.state.inference_sem = asyncio.Semaphore(_INFERENCE_CONCURRENCY)
    app.state.ready = True
    log.info("model loaded on %s", device)
    yield
    app.state.ready = False


app = FastAPI(title="Retinal Disease Classifier", lifespan=lifespan)


class Prediction(BaseModel):
    probabilities: dict[str, float]
    predictions: dict[str, bool]


class HealthResponse(BaseModel):
    status: str


@app.middleware("http")
async def request_logging(request: Request, call_next):
    request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    log.info(
        "rid=%s method=%s path=%s status=%d latency_ms=%.1f",
        request_id, request.method, request.url.path, response.status_code, elapsed_ms,
    )
    response.headers["x-request-id"] = request_id
    return response


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok")


@app.get("/ready", response_model=HealthResponse)
async def ready():
    if not getattr(app.state, "ready", False):
        raise HTTPException(status_code=503, detail="model not ready")
    return HealthResponse(status="ready")


async def _read_image_upload(upload: UploadFile, name: str) -> bytes:
    if upload.content_type not in _ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"{name}: unsupported content type {upload.content_type!r}",
        )
    raw = await upload.read()
    if len(raw) == 0:
        raise HTTPException(status_code=400, detail=f"{name}: empty upload")
    if len(raw) > _MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"{name}: file exceeds {_MAX_IMAGE_BYTES} bytes",
        )
    return raw


def _decode_and_preprocess(raw_bytes: bytes, name: str, transform) -> torch.Tensor:
    try:
        img = Image.open(BytesIO(raw_bytes))
        img.verify()
        img = Image.open(BytesIO(raw_bytes)).convert("RGB")
    except (UnidentifiedImageError, OSError) as e:
        raise HTTPException(status_code=400, detail=f"{name}: cannot decode image ({e})")
    if img.size[0] * img.size[1] > _MAX_PIXELS:
        raise HTTPException(
            status_code=413,
            detail=f"{name}: image too large ({img.size[0]}x{img.size[1]} > {_MAX_PIXELS} px)",
        )
    img = apply_clahe(img)
    return transform(img).unsqueeze(0)


@app.post("/predict", response_model=Prediction)
async def predict(
    left: UploadFile,
    right: UploadFile,
    tta: bool = Query(True, description="Enable test-time augmentation (4 views)"),
):
    left_bytes = await _read_image_upload(left, "left")
    right_bytes = await _read_image_upload(right, "right")
    left_tensor = _decode_and_preprocess(left_bytes, "left", app.state.transform).to(app.state.device)
    right_tensor = _decode_and_preprocess(right_bytes, "right", app.state.transform).to(app.state.device)

    async with app.state.inference_sem:
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
