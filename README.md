# Frankenstein

Hybrid neural network combining EfficientNet-B0 and ResNet-18, connected via trainable adapter.

## Purpose

Cuts two pretrained models in half and fuses them: EfficientNet encoder (1st half) → adapter → ResNet decoder (2nd half residual blocks). Trains adapter-only while freezing encoder/decoder, generating visualizations for activations, training curves, and predictions.

## Setup

```bash
uv sync
uv run python main.py
```

**Requirements:** `efficientnet.pth` and `biasnn.pth` in project root. Uses CIFAR-10 for testing.
