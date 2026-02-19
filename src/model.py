"""
model.py
========
Custom CNN Architecture for Emotion Detection from Facial Images.
Designed for the FER2013 dataset (7 emotion classes, 48x48 grayscale images).

Architecture Overview:
    Input (1x48x48)
        → Block1: Conv2d → BN → ReLU → Conv2d → BN → ReLU → MaxPool → Dropout
        → Block2: Conv2d → BN → ReLU → Conv2d → BN → ReLU → MaxPool → Dropout
        → Block3: Conv2d → BN → ReLU → Conv2d → BN → ReLU → MaxPool → Dropout
        → Block4: Conv2d → BN → ReLU → MaxPool → Dropout
        → AdaptiveAvgPool
        → FC1 → BN → ReLU → Dropout
        → FC2 → BN → ReLU → Dropout
        → Output (7 logits)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    A reusable convolutional block:
        Conv2d → BatchNorm → ReLU → (optional second conv) → MaxPool → Dropout
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_convs: int = 2,
        pool_size: int = 2,
        dropout_rate: float = 0.25,
    ):
        super().__init__()
        layers = []
        for i in range(num_convs):
            c_in = in_channels if i == 0 else out_channels
            layers += [
                nn.Conv2d(c_in, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
        layers += [
            nn.MaxPool2d(kernel_size=pool_size, stride=pool_size),
            nn.Dropout2d(p=dropout_rate),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EmotionCNN(nn.Module):
    """
    Custom CNN for 7-class emotion recognition on 48×48 grayscale images.

    Args:
        num_classes (int): Number of output emotion classes. Default: 7.
        dropout_fc (float): Dropout probability for fully connected layers. Default: 0.5.
        dropout_conv (float): Dropout probability for convolutional blocks. Default: 0.25.
    """

    NUM_EMOTIONS = 7  # FER2013: angry, disgust, fear, happy, sad, surprise, neutral

    def __init__(
        self,
        num_classes: int = 7,
        dropout_fc: float = 0.5,
        dropout_conv: float = 0.25,
    ):
        super().__init__()

        # ── Convolutional Feature Extractor ──────────────────────────────────
        # Input: (B, 1, 48, 48)
        self.block1 = ConvBlock(1, 64, num_convs=2, dropout_rate=dropout_conv)
        # After pool: (B, 64, 24, 24)

        self.block2 = ConvBlock(64, 128, num_convs=2, dropout_rate=dropout_conv)
        # After pool: (B, 128, 12, 12)

        self.block3 = ConvBlock(128, 256, num_convs=2, dropout_rate=dropout_conv)
        # After pool: (B, 256, 6, 6)

        self.block4 = ConvBlock(256, 512, num_convs=1, dropout_rate=dropout_conv)
        # After pool: (B, 512, 3, 3)

        # Global average pool → fixed-size feature vector regardless of spatial dim
        self.gap = nn.AdaptiveAvgPool2d(1)  # (B, 512, 1, 1)

        # ── Classifier Head ───────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_fc),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_fc),
            nn.Linear(128, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        """Kaiming (He) initialization for conv layers; Xavier for linear."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, 1, 48, 48).
        Returns:
            logits: Raw class scores of shape (B, num_classes).
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.gap(x)
        return self.classifier(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities (useful at inference time)."""
        return F.softmax(self.forward(x), dim=1)

    @staticmethod
    def emotion_labels() -> list[str]:
        return ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


# ── Quick sanity-check ────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = EmotionCNN()
    dummy = torch.randn(8, 1, 48, 48)          # batch of 8 grayscale images
    logits = model(dummy)
    print(f"Output shape : {logits.shape}")     # Expected: torch.Size([8, 7])

    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params : {total_params:,}")
    print(f"Trainable    : {trainable:,}")
