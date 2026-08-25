"""
CNN model for urban sound classification.
Designed to be small enough for Cortex-M deployment (INT8 quantizable).

Architecture: 4 conv blocks → global average pooling → FC → 10 classes
Input shape: (batch, 1, 128, 173)  — 1-channel log-mel spectrogram
"""
import torch
import torch.nn as nn


class UrbanSoundCNN(nn.Module):
    """
    Compact CNN for urban sound classification.

    Architecture:
        Conv2d(1→16) → BN → ReLU → MaxPool
        Conv2d(16→32) → BN → ReLU → MaxPool
        Conv2d(32→64) → BN → ReLU → MaxPool
        Conv2d(64→128) → BN → ReLU → GlobalAvgPool
        FC(128→10)

    Total params: ~120K (suitable for INT8 on Cortex-M)
    """

    def __init__(self, num_classes=10):
        super(UrbanSoundCNN, self).__init__()

        self.features = nn.Sequential(
            # Block 1: (1, 128, 173) → (16, 64, 86)
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2: (16, 64, 86) → (32, 32, 43)
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3: (32, 32, 43) → (64, 16, 21)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 4: (64, 16, 21) → (128, 8, 10)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # Global average pooling → (128, 1, 1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # flatten: (batch, 128)
        x = self.classifier(x)
        return x


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    # Quick test
    model = UrbanSoundCNN(num_classes=10)
    total, trainable = count_parameters(model)
    print(f"Model: UrbanSoundCNN")
    print(f"Total parameters:     {total:,}")
    print(f"Trainable parameters: {trainable:,}")

    # Test forward pass
    dummy_input = torch.randn(4, 1, 128, 173)
    output = model(dummy_input)
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")  # Expected: (4, 10)
    print(f"\nModel architecture:\n{model}")
