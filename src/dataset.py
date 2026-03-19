from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.config import Config

# Shared object - parameters are fixed, no state mutated during apply()
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def apply_clahe(img: Image.Image) -> Image.Image:
    lab = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = _clahe.apply(lab[:, :, 0])
    return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))


LABEL_COLS = ["N", "D", "G", "C", "A", "H", "M", "O"]


def make_transforms(
    norm_mean: list[float],
    norm_std: list[float],
    image_size: int,
) -> tuple[transforms.Compose, transforms.Compose]:
    normalize = transforms.Normalize(mean=norm_mean, std=norm_std)
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        normalize,
    ])
    val_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])
    return train_tf, val_tf


def make_raw_val_transform(image_size: int) -> transforms.Compose:
    """Val transform without normalization - for EnsembleModel which normalizes per-backbone internally."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])


class RetinalDataset(Dataset):
    def __init__(self, records: pd.DataFrame, image_dir: Path, transform):
        # records has columns: ID, Left-Fundus, Right-Fundus, N, D, G, C, A, H, M, O
        self.records = records.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row = self.records.iloc[idx]
        left = apply_clahe(Image.open(self.image_dir / row["Left-Fundus"]).convert("RGB"))
        right = apply_clahe(Image.open(self.image_dir / row["Right-Fundus"]).convert("RGB"))
        left = self.transform(left)
        right = self.transform(right)
        labels = row[LABEL_COLS].values.astype(np.float32)
        return left, right, labels


def make_splits(config: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(config.csv_path)

    patient_ids = df["ID"].unique()
    rng = np.random.default_rng(seed=42)
    rng.shuffle(patient_ids)

    n_val = int(len(patient_ids) * config.val_split)
    val_ids = set(patient_ids[:n_val])

    train_df = df[~df["ID"].isin(val_ids)]
    val_df = df[df["ID"].isin(val_ids)]

    return train_df, val_df