"""
train.py
========
Full training pipeline for EmotionCNN on FER2013.

Best practices implemented:
  ✔ Weighted CrossEntropyLoss  (handles class imbalance)
  ✔ AdamW optimizer            (weight decay decoupled from momentum)
  ✔ Cosine Annealing LR + Warmup
  ✔ Early stopping             (patience-based, saves best checkpoint)
  ✔ Gradient clipping          (prevents exploding gradients)
  ✔ Mixed precision (AMP)      (torch.cuda.amp — 2× speedup on GPU)
  ✔ Per-epoch metrics          (Accuracy, Precision, Recall, F1 via sklearn)
  ✔ Confusion matrix           (saved as PNG at end of training)

Usage:
    python src/train.py --data_dir data/raw --epochs 80 --batch_size 64
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from model import EmotionCNN
from dataset import get_dataloaders, EMOTION_LABELS


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        print(f"[Device] CUDA — {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        print("[Device] Apple MPS")
        return torch.device("mps")
    print("[Device] CPU (training will be slow)")
    return torch.device("cpu")


class EarlyStopping:
    """Stop training when validation loss stops improving."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience  = patience
        self.min_delta = min_delta
        self.counter   = 0
        self.best_loss = np.inf

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
            return False           # keep going
        self.counter += 1
        return self.counter >= self.patience   # True → stop


def build_lr_scheduler(optimizer, warmup_epochs: int, total_epochs: int):
    """
    Linear warmup for the first `warmup_epochs`, then cosine annealing.
    Warmup prevents large gradient updates in early training.
    """
    warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs, eta_min=1e-6)
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])


# ─────────────────────────────────────────────────────────────────────────────
# Train / Validate one epoch
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    clip_grad_norm: float = 1.0,
) -> tuple[float, float]:
    """Returns (avg_loss, accuracy)."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)      # faster than zero_grad()

        with autocast(enabled=device.type == "cuda"):
            logits = model(images)
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)
        preds       = logits.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Returns (avg_loss, accuracy, all_preds, all_labels)."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels      = [], []

    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        with autocast(enabled=device.type == "cuda"):
            logits = model(images)
            loss   = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        preds       = logits.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += images.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return (
        total_loss / total,
        correct / total,
        np.array(all_preds),
        np.array(all_labels),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Metrics & Visualization
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(preds: np.ndarray, labels: np.ndarray) -> dict:
    return {
        "accuracy" : accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, average="weighted", zero_division=0),
        "recall"   : recall_score(labels, preds, average="weighted", zero_division=0),
        "f1"       : f1_score(labels, preds, average="weighted", zero_division=0),
    }


def save_confusion_matrix(
    preds: np.ndarray,
    labels: np.ndarray,
    save_path: Path,
    normalize: bool = True,
):
    cm = confusion_matrix(labels, preds)
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    emotion_names = [EMOTION_LABELS[i] for i in range(7)]
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        cm, annot=True, fmt=".2f" if normalize else "d",
        xticklabels=emotion_names, yticklabels=emotion_names,
        cmap="Blues", ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("Confusion Matrix (Normalized)" if normalize else "Confusion Matrix", fontsize=14)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[Saved] Confusion matrix → {save_path}")


def save_training_curves(history: dict, save_path: Path):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, history["train_loss"], label="Train Loss")
    axes[0].plot(epochs, history["val_loss"],   label="Val Loss")
    axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], label="Train Acc")
    axes[1].plot(epochs, history["val_acc"],   label="Val Acc")
    axes[1].set_title("Accuracy"); axes[1].set_xlabel("Epoch")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[Saved] Training curves → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main Training Loop
# ─────────────────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()

    # ── Data ─────────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader, class_weights = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_weighted_sampler=True,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = EmotionCNN(num_classes=7, dropout_fc=args.dropout_fc, dropout_conv=args.dropout_conv)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Trainable parameters: {total_params:,}")

    # ── Loss — weighted to address class imbalance ────────────────────────────
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=0.1)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )

    # ── LR Schedule: warmup → cosine annealing ───────────────────────────────
    scheduler = build_lr_scheduler(optimizer, warmup_epochs=5, total_epochs=args.epochs)

    # ── AMP Scaler ────────────────────────────────────────────────────────────
    scaler = GradScaler(enabled=device.type == "cuda")

    # ── Early Stopping ────────────────────────────────────────────────────────
    stopper    = EarlyStopping(patience=args.patience)
    best_acc   = 0.0
    best_ckpt  = output_dir / "best_model.pth"

    history = {k: [] for k in ["train_loss", "val_loss", "train_acc", "val_acc"]}

    print(f"\n{'='*60}")
    print(f"  Starting training  |  epochs={args.epochs}  |  lr={args.lr}")
    print(f"{'='*60}\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )

        # Validate
        val_loss, val_acc, val_preds, val_labels = evaluate(
            model, val_loader, criterion, device
        )

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - t0
        print(
            f"Epoch [{epoch:03d}/{args.epochs}]  "
            f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}  "
            f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f}  "
            f"LR: {current_lr:.2e}  ({elapsed:.1f}s)"
        )

        # Save best checkpoint
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "epoch":           epoch,
                    "model_state":     model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_acc":         val_acc,
                    "val_loss":        val_loss,
                    "args":            vars(args),
                },
                best_ckpt,
            )
            print(f"  ✔ New best val_acc={val_acc:.4f} — checkpoint saved.")

        # Early stopping
        if stopper(val_loss):
            print(f"\n[Early Stopping] No val_loss improvement for {args.patience} epochs.")
            break

    # ── Post-training ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Training complete  |  Best Val Acc: {best_acc:.4f}")
    print(f"{'='*60}\n")

    save_training_curves(history, output_dir / "training_curves.png")

    # ── Final evaluation on test set ─────────────────────────────────────────
    checkpoint = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    print(f"[Eval] Loaded best checkpoint from epoch {checkpoint['epoch']}")

    test_loss, test_acc, test_preds, test_labels = evaluate(
        model, test_loader, criterion, device
    )
    metrics = compute_metrics(test_preds, test_labels)

    print("\n── Test Set Results ─────────────────────────────────────────")
    print(f"  Loss      : {test_loss:.4f}")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1 Score  : {metrics['f1']:.4f}")
    print("\n── Per-class Report ─────────────────────────────────────────")
    emotion_names = [EMOTION_LABELS[i] for i in range(7)]
    print(classification_report(test_labels, test_preds, target_names=emotion_names))

    save_confusion_matrix(test_preds, test_labels, output_dir / "confusion_matrix.png")

    # Save predictions for failure analysis
    np.save(output_dir / "test_preds.npy",  test_preds)
    np.save(output_dir / "test_labels.npy", test_labels)
    print(f"[Saved] Predictions → {output_dir}/test_preds.npy")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train EmotionCNN on FER2013")
    p.add_argument("--data_dir",     type=str,   default="data/raw",   help="Root dir with train/ and test/ subfolders")
    p.add_argument("--output_dir",   type=str,   default="outputs",    help="Directory for checkpoints & plots")
    p.add_argument("--epochs",       type=int,   default=80,           help="Max training epochs")
    p.add_argument("--batch_size",   type=int,   default=64,           help="Batch size")
    p.add_argument("--lr",           type=float, default=1e-3,         help="Peak learning rate (AdamW)")
    p.add_argument("--weight_decay", type=float, default=1e-4,         help="AdamW weight decay")
    p.add_argument("--dropout_fc",   type=float, default=0.5,          help="Dropout for FC layers")
    p.add_argument("--dropout_conv", type=float, default=0.25,         help="Dropout for conv blocks")
    p.add_argument("--patience",     type=int,   default=12,           help="Early stopping patience")
    p.add_argument("--num_workers",  type=int,   default=4,            help="DataLoader workers")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
