from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.config import Config

LABEL_COLS = ["N", "D", "G", "C", "A", "H", "M", "O"]

TRAIN_TRANSFORMS = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

VAL_TRANSFORMS = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
        left = Image.open(self.image_dir / row["Left-Fundus"]).convert("RGB")
        right = Image.open(self.image_dir / row["Right-Fundus"]).convert("RGB")
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