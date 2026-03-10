from pathlib import Path

from pydantic_settings import BaseSettings


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

    # Training
    batch_size: int = 64
    batch_size_finetune: int = 56
    lr_head: float = 1e-3
    lr_finetune: float = 3e-4
    weight_decay: float = 1e-2
    epochs_frozen: int = 5
    epochs_unfrozen: int = 25
    val_split: float = 0.2
    patience: int = 5

    # Hardware
    device: str = "cuda"
    num_workers: int = 8

    model_config = {"env_file": ".env"}