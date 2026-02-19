"""
analyze_failures.py
===================
Failure Case Analysis — visualize images the model predicted incorrectly
alongside their ground truth labels and predicted labels.

Helps diagnose:
  • Which emotion pairs are most confused (e.g. Fear↔Surprise)
  • Whether failures cluster around low confidence or near-50/50 softmax
  • Common failure patterns (extreme lighting, occlusion, profile views)

Usage:
    python analyze_failures.py \
        --csv fer2013.csv \
        --checkpoint outputs/best_model.pth \
        --num_images 30 \
        --split PrivateTest \
        --output_dir outputs/failure_analysis
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from dataset import FER2013Dataset, get_val_test_transforms, EMOTION_LABELS
from model import EmotionCNN

import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Core analysis
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def collect_failures(
    model: torch.nn.Module,
    dataset: FER2013Dataset,
    device: torch.device,
    batch_size: int = 64,
) -> list[dict]:
    """
    Run full inference on `dataset` and collect all misclassified samples.

    Returns:
        List of dicts with keys:
            idx, image_tensor, true_label, pred_label, confidence, probs
    """
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    model.eval()

    failures = []
    sample_idx = 0

    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        probs  = F.softmax(logits, dim=1).cpu()
        preds  = probs.argmax(dim=1)

        for i in range(images.size(0)):
            true  = labels[i].item()
            pred  = preds[i].item()
            if pred != true:
                failures.append({
                    "idx"         : sample_idx + i,
                    "image_tensor": images[i].cpu(),
                    "true_label"  : true,
                    "pred_label"  : pred,
                    "confidence"  : probs[i, pred].item(),
                    "probs"       : probs[i].numpy(),
                })

        sample_idx += images.size(0)

    return failures


def denormalize(tensor: torch.Tensor, mean: float = 0.5071, std: float = 0.2551) -> np.ndarray:
    """Reverse normalization → clip to [0,1] → numpy for matplotlib."""
    img = tensor.squeeze().numpy() * std + mean
    return np.clip(img, 0, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Visualization helpers
# ─────────────────────────────────────────────────────────────────────────────

def plot_failure_grid(
    failures: list[dict],
    num_images: int,
    save_path: Path,
    sort_by_confidence: bool = True,
):
    """
    Grid of misclassified images.
    Title = "True: X | Pred: Y (conf XX%)"
    Background colour = red for wrong, subtle green box for ground-truth emotion.
    """
    if sort_by_confidence:
        # Show the most confident wrong predictions first — worst failures
        failures = sorted(failures, key=lambda x: -x["confidence"])

    subset = failures[:num_images]
    ncols  = 6
    nrows  = (len(subset) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.8, nrows * 3.2))
    axes = axes.flatten()

    for ax in axes:
        ax.axis("off")

    for i, f in enumerate(subset):
        ax = axes[i]
        img = denormalize(f["image_tensor"])
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(
            f"True: {EMOTION_LABELS[f['true_label']]}\n"
            f"Pred: {EMOTION_LABELS[f['pred_label']]} ({f['confidence']*100:.1f}%)",
            fontsize=7,
            color="darkred",
        )
        for spine in ax.spines.values():
            spine.set_edgecolor("crimson")
            spine.set_linewidth(2)
        ax.axis("on")
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(
        f"Failure Cases — Top {len(subset)} Most Confident Wrong Predictions",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] Failure grid → {save_path}")


def plot_confusion_pairs(failures: list[dict], save_path: Path, top_n: int = 10):
    """Bar chart of the most common (true→pred) confusion pairs."""
    from collections import Counter

    pairs  = [(EMOTION_LABELS[f["true_label"]], EMOTION_LABELS[f["pred_label"]]) for f in failures]
    counts = Counter(pairs).most_common(top_n)

    labels = [f"{t} → {p}" for (t, p), _ in counts]
    values = [c for _, c in counts]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(labels[::-1], values[::-1], color="steelblue", edgecolor="navy")
    ax.bar_label(bars, padding=3, fontsize=9)
    ax.set_xlabel("Count")
    ax.set_title(f"Top {top_n} Confusion Pairs (True → Predicted)", fontsize=13)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[Saved] Confusion pairs → {save_path}")


def plot_confidence_distribution(failures: list[dict], save_path: Path):
    """Histogram of model confidence on wrong predictions."""
    confs = [f["confidence"] for f in failures]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(confs, bins=20, color="tomato", edgecolor="darkred", alpha=0.8)
    ax.axvline(np.mean(confs), color="navy", linestyle="--", label=f"Mean: {np.mean(confs):.2f}")
    ax.set_xlabel("Model Confidence on Wrong Prediction")
    ax.set_ylabel("Count")
    ax.set_title("Confidence Distribution for Failure Cases")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[Saved] Confidence distribution → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # Load model
    model     = EmotionCNN()
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model     = model.to(device)
    print(f"[Model] Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")

    # Load split
    df = pd.read_csv(args.csv)
    split_df = df[df["Usage"] == args.split].reset_index(drop=True)
    print(f"[Data] {args.split} split — {len(split_df):,} samples")
    dataset = FER2013Dataset(split_df, transform=get_val_test_transforms())

    # Collect failures
    failures = collect_failures(model, dataset, device)
    print(f"\n[Analysis] Total failures: {len(failures)} / {len(dataset)} "
          f"({len(failures)/len(dataset)*100:.1f}% error rate)")

    # Visualise
    plot_failure_grid(
        failures, num_images=args.num_images,
        save_path=output_dir / "failure_grid.png",
    )
    plot_confusion_pairs(failures, save_path=output_dir / "confusion_pairs.png")
    plot_confidence_distribution(failures, save_path=output_dir / "confidence_dist.png")

    # Save summary CSV
    summary = [{
        "sample_idx"  : f["idx"],
        "true_emotion": EMOTION_LABELS[f["true_label"]],
        "pred_emotion": EMOTION_LABELS[f["pred_label"]],
        "confidence"  : round(f["confidence"], 4),
    } for f in failures]
    pd.DataFrame(summary).to_csv(output_dir / "failures.csv", index=False)
    print(f"[Saved] Failure summary CSV → {output_dir}/failures.csv")

    print(f"\nAll failure analysis artefacts saved to → {output_dir}/")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Failure Case Analysis for EmotionCNN")
    p.add_argument("--csv",        type=str, required=True,                    help="Path to fer2013.csv")
    p.add_argument("--checkpoint", type=str, default="outputs/best_model.pth", help="Path to model checkpoint")
    p.add_argument("--num_images", type=int, default=30,                       help="Number of failure images to display")
    p.add_argument("--split",      type=str, default="PrivateTest",            choices=["Training", "PublicTest", "PrivateTest"])
    p.add_argument("--output_dir", type=str, default="outputs/failure_analysis")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
