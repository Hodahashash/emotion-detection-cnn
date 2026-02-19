"""
dataset.py
==========
FER2013 Dataset loader with train/val/test splits and aggressive data augmentation
to combat the class imbalance and overfitting typical of the FER2013 dataset.

FER2013 CSV format:
    emotion,pixels,Usage
    0,"70 80 82 ...",Training
    ...

Usage:
    from dataset import get_dataloaders
    train_loader, val_loader, test_loader, class_weights = get_dataloaders("fer2013.csv")
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from pathlib import Path


# ── Label mapping ─────────────────────────────────────────────────────────────
EMOTION_LABELS = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral",
}

IMAGE_SIZE = 48  # FER2013 native resolution


# ── Transforms ────────────────────────────────────────────────────────────────

def get_train_transforms() -> transforms.Compose:
    """
    Augmentation pipeline for training.
    Chosen to simulate realistic facial variation while preserving emotion cues.
    """
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),     # subtle shift
            scale=(0.85, 1.15),       # zoom-in / zoom-out
            shear=5,
        ),
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.2
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071], std=[0.2551]),  # computed on FER2013 train set
        transforms.RandomErasing(p=0.15, scale=(0.02, 0.08)),  # occlusion robustness
    ])


def get_val_test_transforms() -> transforms.Compose:
    """Deterministic pipeline for validation and test sets — no augmentation."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071], std=[0.2551]),
    ])


# ── Dataset class ─────────────────────────────────────────────────────────────

class FER2013Dataset(Dataset):
    """
    PyTorch Dataset for the FER2013 CSV file.

    Args:
        dataframe (pd.DataFrame): Rows for this split (Training / PublicTest / PrivateTest).
        transform (callable, optional): Transform applied to each PIL image.
    """

    def __init__(self, dataframe: pd.DataFrame, transform=None):
        self.labels  = dataframe["emotion"].values.astype(np.int64)
        self.pixels  = dataframe["pixels"].values          # space-separated strings
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        # Parse pixel string → numpy array → PIL image
        pixel_values = np.fromstring(self.pixels[idx], sep=" ", dtype=np.uint8)
        image = Image.fromarray(pixel_values.reshape(IMAGE_SIZE, IMAGE_SIZE), mode="L")

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label


# ── DataLoader factory ────────────────────────────────────────────────────────

def compute_class_weights(labels: np.ndarray, num_classes: int = 7) -> torch.Tensor:
    """
    Compute inverse-frequency class weights to pass to CrossEntropyLoss.
    This helps with FER2013's severe class imbalance (Disgust ≈ 547 vs Happy ≈ 8989).
    """
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * num_classes   # normalize
    return torch.tensor(weights, dtype=torch.float32)


def get_dataloaders(
    csv_path: str,
    batch_size: int = 64,
    num_workers: int = 4,
    use_weighted_sampler: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    """
    Build train / val / test DataLoaders from a FER2013 CSV file.

    Args:
        csv_path:              Path to fer2013.csv.
        batch_size:            Samples per batch.
        num_workers:           DataLoader worker processes.
        use_weighted_sampler:  Oversample minority classes during training.

    Returns:
        (train_loader, val_loader, test_loader, class_weights)
    """
    df = pd.read_csv(csv_path)

    train_df = df[df["Usage"] == "Training"].reset_index(drop=True)
    val_df   = df[df["Usage"] == "PublicTest"].reset_index(drop=True)
    test_df  = df[df["Usage"] == "PrivateTest"].reset_index(drop=True)

    print(f"[Dataset] Train: {len(train_df):,}  |  Val: {len(val_df):,}  |  Test: {len(test_df):,}")

    train_ds = FER2013Dataset(train_df, transform=get_train_transforms())
    val_ds   = FER2013Dataset(val_df,   transform=get_val_test_transforms())
    test_ds  = FER2013Dataset(test_df,  transform=get_val_test_transforms())

    class_weights = compute_class_weights(train_df["emotion"].values)

    # ── WeightedRandomSampler: ensures every batch sees balanced classes ──────
    if use_weighted_sampler:
        sample_weights = class_weights[train_df["emotion"].values]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_ds),
            replacement=True,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, class_weights


# ── Quick smoke-test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    csv_path = sys.argv[1] if len(sys.argv) > 1 else "fer2013.csv"
    if not Path(csv_path).exists():
        print(f"CSV not found at '{csv_path}'. Pass path as first argument.")
        sys.exit(1)

    train_loader, val_loader, test_loader, cw = get_dataloaders(csv_path, batch_size=32)
    imgs, labels = next(iter(train_loader))
    print(f"Batch shape : {imgs.shape}")     # (32, 1, 48, 48)
    print(f"Label dtype : {labels.dtype}")
    print(f"Class weights: {cw.numpy().round(4)}")
