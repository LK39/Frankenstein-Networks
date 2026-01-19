# Frankenstein

Hybrid neural network combining EfficientNet-B0 and ResNet-18, connected via trainable adapter.

## Purpose

Cuts two pretrained models in half and fuses them: EfficientNet encoder (1st half) → adapter → ResNet decoder (2nd half residual blocks). Trains adapter-only while freezing encoder/decoder, generating visualizations for activations, training curves, and predictions.

## Setup

```bash
uv sync
uv run python main.py
```

**Requirements:** `biasnn.pth` in project root, else ResNet-18 will download. Uses CIFAR-10 for testing. 

## Baseline Output 

```bash

>>> EfficientNet-B0

==================================================
EfficientNet-B0 Summary
==================================================
Total params: 4,020,358
Trainable params: 4,020,358
Non-trainable params: 0
==================================================

EfficientNet-B0 - Architecture Breakdown:
----------------------------------------------------------------------
  [0] Sequential                        4,007,548 params
  [1] AdaptiveAvgPool2d                         0 params
  [2] Sequential                           12,810 params
----------------------------------------------------------------------
Head Epoch 1, Loss: 0.9074
Head Epoch 2, Loss: 0.7057
Head Epoch 3, Loss: 0.6777
Head Epoch 4, Loss: 0.6601
Head Epoch 5, Loss: 0.6536
Head Epoch 6, Loss: 0.6504
Head Epoch 7, Loss: 0.6418
Head Epoch 8, Loss: 0.6411
Head Epoch 9, Loss: 0.6402
Head Epoch 10, Loss: 0.6479
Head Epoch 11, Loss: 0.6422
Head Epoch 12, Loss: 0.6428
Head Epoch 13, Loss: 0.6424
Head Epoch 14, Loss: 0.6444
Head Epoch 15, Loss: 0.6363
Accuracy: 0.8160

>>> BIASNN (ResNet-18)

==================================================
BIASNN/ResNet-18 Summary
==================================================
Total params: 11,181,642
Trainable params: 11,181,642
Non-trainable params: 0
==================================================

BIASNN/ResNet-18 - Architecture Breakdown:
----------------------------------------------------------------------
  [0] Conv2d                                9,408 params
  [1] BatchNorm2d                             128 params
  [2] ReLU                                      0 params
  [3] MaxPool2d                                 0 params
  [4] Sequential                          147,968 params
  [5] Sequential                          525,568 params
  [6] Sequential                        2,099,712 params
  [7] Sequential                        8,393,728 params
  [8] AdaptiveAvgPool2d                         0 params
  [9] Linear                                5,130 params
----------------------------------------------------------------------
Head Epoch 1, Loss: 0.8186
Head Epoch 2, Loss: 0.6205
Head Epoch 3, Loss: 0.5902
Head Epoch 4, Loss: 0.5774
Head Epoch 5, Loss: 0.5684
Head Epoch 6, Loss: 0.5607
Head Epoch 7, Loss: 0.5602
Head Epoch 8, Loss: 0.5538
Head Epoch 9, Loss: 0.5510
Head Epoch 10, Loss: 0.5431
Head Epoch 11, Loss: 0.5445
Head Epoch 12, Loss: 0.5407
Head Epoch 13, Loss: 0.5427
Head Epoch 14, Loss: 0.5413
Head Epoch 15, Loss: 0.5400
Accuracy: 0.8049

>>> Creating Activation Visualizations
✓ Saved activation plots to outputs/

>>> Hybrid Model
Using EfficientNet features 0-4 (richer mid-level features) + ResNet layers 4-7 (residual decoder)

Hybrid Construction:
  Model A: 3 layers, using first 1
  Model B: 10 layers, using last 4
  Decoder: using 2 convolutional blocks, will add pooling+classifier
  Encoder output: torch.Size([1, 1280, 7, 7])
  Decoder first layer: Conv2d
    Expects Conv2d input: in_channels=128
  Adapter created: Conv2d
  After adapter: torch.Size([1, 128, 7, 7])
  After decoder: torch.Size([1, 512, 2, 2])
  ✓ Pipeline validation successful!
  Added pooling+classifier head: 512→10
HybridModel: adapter created: Conv2d, params=163968

==================================================
Hybrid Model Summary
==================================================
Total params: 14,670,086
Trainable params: 14,670,086
Non-trainable params: 0

Encoder: 4,007,548 | Adapter: 163,968 | Decoder: 10,498,570
==================================================

Hybrid Architecture Breakdown:
Encoder layers:
  [0] Sequential                        4,007,548 params

Decoder layers:
  [0] Sequential                       10,493,440 params
  [1] AdaptiveAvgPool2d                         0 params
  [2] Flatten                                   0 params
  [3] Linear                                5,130 params

Adapter: Conv2d

>>> Training Adapter Only (15 epochs, lr=1e-3)
Epoch 1, Loss: 1.6134
Epoch 2, Loss: 1.2603
Epoch 3, Loss: 1.1588
Epoch 4, Loss: 1.1002
Epoch 5, Loss: 1.0667
Epoch 6, Loss: 1.0353
Epoch 7, Loss: 1.0106
Epoch 8, Loss: 0.9982
Epoch 9, Loss: 0.9806
Epoch 10, Loss: 0.9718
Epoch 11, Loss: 0.9535
Epoch 12, Loss: 0.9425
Epoch 13, Loss: 0.9360
Epoch 14, Loss: 0.9290
Epoch 15, Loss: 0.9224
Adapter-Only Accuracy: 0.7696

==================================================
Final Results: Eff=0.8160 | Bias=0.8049 | Hybrid=0.7696
==================================================
```
