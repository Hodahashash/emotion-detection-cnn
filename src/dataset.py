"""
dataset.py
==========
FER2013 Dataset loader for image-folder format with train/val/test splits
and aggressive data augmentation to combat class imbalance and overfitting.

Expected folder structure:
    data/raw/
    ├── train/
    │   ├── angry/
    │   ├── disgust/
    │   ├── fear/
    │   ├── happy/
    │   ├── neutral/
    │   ├── sad/
    │   └── surprise/
    └── test/
        ├── angry/
        ├── disgust/
        └── ...

Usage:
    from dataset import get_dataloaders
    train_loader, val_loader, test_loader, class_weights = get_dataloaders("data/raw")
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from pathlib import Path


# ── Label mapping (alphabetical — matches ImageFolder auto-assignment) ────────
EMOTION_LABELS = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprise",
}

IMAGE_SIZE = 48  # FER2013 native resolution


# ── Transforms ────────────────────────────────────────────────────────────────

def get_train_transforms() -> transforms.Compose:
    """
    Augmentation pipeline for training.
    Chosen to simulate realistic facial variation while preserving emotion cues.
    Grayscale conversion is included to handle RGB-saved images.
    """
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
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
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071], std=[0.2551]),
    ])


# ── DataLoader factory ────────────────────────────────────────────────────────

def compute_class_weights(dataset: ImageFolder, num_classes: int = 7) -> torch.Tensor:
    """
    Compute inverse-frequency class weights to pass to CrossEntropyLoss.
    This helps with FER2013's severe class imbalance (Disgust ≈ 547 vs Happy ≈ 8989).
    """
    labels = np.array([label for _, label in dataset.samples])
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * num_classes   # normalize
    return torch.tensor(weights, dtype=torch.float32)


def get_dataloaders(
    data_dir: str,
    batch_size: int = 64,
    num_workers: int = 4,
    val_split: float = 0.15,
    use_weighted_sampler: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    """
    Build train / val / test DataLoaders from an image folder.

    Args:
        data_dir:              Root directory containing train/ and test/ subfolders.
        batch_size:            Samples per batch.
        num_workers:           DataLoader worker processes.
        val_split:             Fraction of training data to use for validation.
        use_weighted_sampler:  Oversample minority classes during training.

    Returns:
        (train_loader, val_loader, test_loader, class_weights)
    """
    data_dir = Path(data_dir)
    train_dir = data_dir / "train"
    test_dir  = data_dir / "test"

    # Load full training set (with augmentation) and a clean copy for val split
    full_train_ds = ImageFolder(train_dir, transform=get_train_transforms())
    full_val_ds   = ImageFolder(train_dir, transform=get_val_test_transforms())
    test_ds       = ImageFolder(test_dir,  transform=get_val_test_transforms())

    # ── Train / Val split ────────────────────────────────────────────────────
    n_total = len(full_train_ds)
    n_val   = int(n_total * val_split)
    n_train = n_total - n_val

    indices      = torch.randperm(n_total).tolist()
    train_idx    = indices[:n_train]
    val_idx      = indices[n_train:]

    train_ds = Subset(full_train_ds, train_idx)   # augmented
    val_ds   = Subset(full_val_ds,   val_idx)     # clean (no augmentation)

    print(f"[Dataset] Train: {len(train_ds):,}  |  Val: {len(val_ds):,}  |  Test: {len(test_ds):,}")
    print(f"[Dataset] Classes: {full_train_ds.classes}")

    class_weights = compute_class_weights(full_train_ds)

    # ── WeightedRandomSampler: ensures every batch sees balanced classes ──────
    if use_weighted_sampler:
        all_labels     = np.array([full_train_ds.targets[i] for i in train_idx])
        sample_weights = class_weights[all_labels]
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

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/raw"
    if not Path(data_dir).exists():
        print(f"Data directory not found at '{data_dir}'. Pass path as first argument.")
        sys.exit(1)

    train_loader, val_loader, test_loader, cw = get_dataloaders(data_dir, batch_size=32)
    imgs, labels = next(iter(train_loader))
    print(f"Batch shape  : {imgs.shape}")       # (32, 1, 48, 48)
    print(f"Label dtype  : {labels.dtype}")
    print(f"Class weights: {cw.numpy().round(4)}")
