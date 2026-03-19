from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings

# Backbones whose pretrained weights expect [-1, 1] input (Inception-style normalization)
_INCEPTION_NORM: tuple[list[float], list[float]] = ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
_BACKBONE_NORM: dict[str, tuple[list[float], list[float]]] = {
    "inception_resnet_v2": _INCEPTION_NORM,
    "inception_v3": _INCEPTION_NORM,
    "inception_v4": _INCEPTION_NORM,
}


class Config(BaseSettings):
    # Paths
    csv_path: Path = Path("data/archive/full_df.csv")
    image_dir: Path = Path("data/archive/ODIR-5K/ODIR-5K/Training Images")
    checkpoint_dir: Path = Path("checkpoints")

    # Labels - order matches N,D,G,C,A,H,M,O columns in full_df.csv
    labels: list[str] = ["N", "D", "G", "C", "A", "H", "M", "O"]

    # Model
    backbone: str = "efficientnet_b4"
    image_size: int = 448
    dropout: float = 0.5
    norm_mean: list[float] = [0.485, 0.456, 0.406]
    norm_std: list[float] = [0.229, 0.224, 0.225]

    # Training
    batch_size: int = 64
    batch_size_finetune: int = 56
    lr_head: float = 1e-3
    lr_finetune: float = 1e-4
    weight_decay: float = 1e-3
    epochs_frozen: int = 5
    epochs_unfrozen: int = 25
    val_split: float = 0.2
    patience: int = 7

    # Hardware
    device: str = "cuda"
    num_workers: int = 8

    model_config = {"env_file": ".env"}

    @model_validator(mode="after")
    def _apply_backbone_defaults(self) -> "Config":
        if "checkpoint_dir" not in self.model_fields_set:
            self.checkpoint_dir = Path("checkpoints") / self.backbone
        if self.backbone in _BACKBONE_NORM and "norm_mean" not in self.model_fields_set:
            self.norm_mean, self.norm_std = _BACKBONE_NORM[self.backbone]
        return self