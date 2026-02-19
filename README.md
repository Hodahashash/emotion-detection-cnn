# 🧠 Emotion Detection from Facial Images Using CNNs

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![Dataset](https://img.shields.io/badge/Dataset-FER2013-orange)](https://www.kaggle.com/datasets/msambare/fer2013)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> A deep learning system that classifies human facial expressions into 7 emotion categories using a custom Convolutional Neural Network trained from scratch on the FER2013 dataset.

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Architecture](#-architecture)
- [Dataset](#-dataset)
- [Results](#-results)
- [Project Structure](#-project-structure)
- [How to Run](#-how-to-run)
- [Failure Analysis](#-failure-case-analysis)
- [Key Design Decisions](#-key-design-decisions)

---

## 📌 Project Overview

Emotion recognition is a core challenge in affective computing, human-computer interaction, and accessibility technology. This project demonstrates:

- Building a **production-quality CNN from scratch** (no pretrained weights) using PyTorch
- Handling **real-world dataset challenges**: class imbalance, noisy labels, low resolution (48×48)
- Full ML lifecycle: dataset loading → augmentation → training → evaluation → failure analysis
- Best practices including mixed-precision training, learning rate warm-up, early stopping, and weighted loss

**Emotion Classes (FER2013):** Angry · Disgust · Fear · Happy · Sad · Surprise · Neutral

---

## 🏗️ Architecture

```
Input (1 × 48 × 48)
│
├─── ConvBlock 1: [Conv2d(1→64) → BN → ReLU] × 2 → MaxPool2d → Dropout2d(0.25)
│    Output: 64 × 24 × 24
│
├─── ConvBlock 2: [Conv2d(64→128) → BN → ReLU] × 2 → MaxPool2d → Dropout2d(0.25)
│    Output: 128 × 12 × 12
│
├─── ConvBlock 3: [Conv2d(128→256) → BN → ReLU] × 2 → MaxPool2d → Dropout2d(0.25)
│    Output: 256 × 6 × 6
│
├─── ConvBlock 4: [Conv2d(256→512) → BN → ReLU] → MaxPool2d → Dropout2d(0.25)
│    Output: 512 × 3 × 3
│
├─── AdaptiveAvgPool2d → (512 × 1 × 1)
│
└─── Classifier Head
     Linear(512→256) → BN → ReLU → Dropout(0.5)
     Linear(256→128) → BN → ReLU → Dropout(0.5)
     Linear(128→7)  → Logits
```

**Key architectural choices:**
| Component | Choice | Rationale |
|---|---|---|
| Normalization | BatchNorm after every Conv | Stabilizes training, enables higher LR |
| Pooling | MaxPool2d (spatial) + AdaptiveAvgPool (global) | Reduces spatial dims; last pool removes FC dependency on input size |
| Regularization | Dropout2d (conv) + Dropout (FC) | Prevents co-adaptation of feature maps and neurons |
| Init | Kaiming (conv), Xavier (linear) | Matched to ReLU activation gain |

**Total parameters: ~4.2M**

---

## 📊 Dataset

**FER2013** — 35,887 grayscale 48×48 facial images across 7 classes.

| Split              | Samples |
| ------------------ | ------- |
| Training           | 28,709  |
| PublicTest (val)   | 3,589   |
| PrivateTest (test) | 3,589   |

> ⚠️ **Class Imbalance:** Disgust (547) vs Happy (8,989) — a 16× imbalance handled via `WeightedRandomSampler` + `CrossEntropyLoss(weight=...)`.

### Data Augmentation (Training Only)

- Random horizontal flip (p=0.5)
- Random rotation ±15°
- Random affine: translate ±10%, scale 85–115%, shear ±5°
- Gaussian blur (p=0.2)
- Random erasing / cutout (p=0.15)
- Normalize: mean=0.5071, std=0.2551

---

## 📈 Results

| Metric             | Score   |
| ------------------ | ------- |
| Test Accuracy      | ~66–68% |
| Weighted Precision | ~0.67   |
| Weighted Recall    | ~0.67   |
| Weighted F1        | ~0.67   |

> 🗒️ _State-of-the-art on FER2013 with custom CNNs is ~68–72%; pretrained transformers push to ~75%+. Results may vary based on hardware and random seed._

Training outputs saved to `outputs/`:

- `best_model.pth` — Best validation checkpoint
- `confusion_matrix.png` — Normalized per-class confusion matrix
- `training_curves.png` — Loss & accuracy over epochs

---

## 🗂️ Project Structure

```
emotion-detection-cnn/
│
├── model.py               # CNN architecture (EmotionCNN)
├── dataset.py             # FER2013 Dataset class + augmentation + DataLoader factory
├── train.py               # Full training loop with evaluation
├── analyze_failures.py    # Failure case visualization & analysis
│
├── outputs/               # Generated at runtime
│   ├── best_model.pth
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   └── failure_analysis/
│       ├── failure_grid.png
│       ├── confusion_pairs.png
│       ├── confidence_dist.png
│       └── failures.csv
│
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

### 1. Clone and Install Dependencies

```bash
git clone https://github.com/YOUR_USERNAME/emotion-detection-cnn.git
cd emotion-detection-cnn
pip install -r requirements.txt
```

### 2. Download the Dataset

```bash
# Option A: Kaggle CLI
kaggle datasets download -d msambare/fer2013
unzip fer2013.zip

# Option B: Download manually from
# https://www.kaggle.com/datasets/msambare/fer2013
# Place fer2013.csv in the project root.
```

### 3. Train

```bash
# Default settings (80 epochs, batch 64, lr=1e-3)
python train.py --csv fer2013.csv

# Custom settings
python train.py \
    --csv fer2013.csv \
    --epochs 100 \
    --batch_size 128 \
    --lr 5e-4 \
    --output_dir runs/exp1
```

### 4. Run Failure Analysis

```bash
python analyze_failures.py \
    --csv fer2013.csv \
    --checkpoint outputs/best_model.pth \
    --num_images 30 \
    --output_dir outputs/failure_analysis
```

### 5. Quick Architecture Check

```bash
python model.py    # prints output shape and param count
python dataset.py fer2013.csv  # smoke-tests the DataLoader
```

---

## 🔍 Failure Case Analysis

The `analyze_failures.py` script produces:

1. **`failure_grid.png`** — Grid of misclassified images with true vs predicted labels and model confidence. Sorted by highest-confidence wrong predictions (i.e. the worst failures).

2. **`confusion_pairs.png`** — Bar chart of the most common true→predicted confusion pairs. Expect high `Fear→Sad` and `Disgust→Angry` confusion — emotions that share facial muscle patterns.

3. **`confidence_dist.png`** — Histogram of model confidence on wrong predictions. A spike near 1.0 indicates overconfident wrong predictions — a target for calibration.

4. **`failures.csv`** — Machine-readable table for programmatic analysis.

---

## 🎛️ Key Design Decisions

| Decision                      | Detail                                                                    |
| ----------------------------- | ------------------------------------------------------------------------- |
| **No pretrained weights**     | Full training from scratch demonstrates understanding of CNN fundamentals |
| **Label smoothing (ε=0.1)**   | Prevents overconfident predictions, improves calibration                  |
| **AdamW + weight decay**      | Weight decay is decoupled from the adaptive gradient, unlike vanilla Adam |
| **Warmup + Cosine Annealing** | Warmup avoids large early updates; cosine annealing finds sharp minima    |
| **Mixed precision (AMP)**     | ~2× faster training on CUDA GPUs with no accuracy cost                    |
| **WeightedRandomSampler**     | Ensures minority class (Disgust) appears in every batch                   |
| **Kaiming init**              | Empirically shown to improve convergence for ReLU networks                |

---

## 📦 Requirements

```
torch>=2.0.0
torchvision>=0.15.0
numpy
pandas
scikit-learn
matplotlib
seaborn
Pillow
```

Install:

```bash
pip install torch torchvision numpy pandas scikit-learn matplotlib seaborn Pillow
```

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

_Built with ❤️ using PyTorch — no pretrained weights, no shortcuts._
