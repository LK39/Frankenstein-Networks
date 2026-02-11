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


================================================================================
HYBRID MODEL OPTIMIZATION USING LAYER MATCHING FRAMEWORK
================================================================================

Loading models...
✓ Model A (efficientnet.pth): 3 top-level layers
✓ Model B (biasnn.pth): 10 top-level layers

================================================================================
ANALYZING ALL CUT POINTS
================================================================================  

Analyzing: A[0:1] → B[0:]
  Layer A shape: (1280, 320)
  Layer B shape: (64, 147)
  Overall Match: 47.8%
  Metrics: AAS=0.911, CUR=0.469, RM=0.124, GFE=0.000

Analyzing: A[0:1] → B[1:]
  ⚠ No extractable weights, skipping

Analyzing: A[0:1] → B[2:]
  ⚠ No extractable weights, skipping

Analyzing: A[0:1] → B[3:]
  ⚠ No extractable weights, skipping

Analyzing: A[0:1] → B[4:]
  Layer A shape: (1280, 320)
  Layer B shape: (64, 576)
  Overall Match: 48.1%
  Metrics: AAS=0.954, CUR=0.422, RM=0.041, GFE=0.104

Analyzing: A[0:1] → B[5:]
  Layer A shape: (1280, 320)
  Layer B shape: (128, 576)
  Overall Match: 47.0%
  Metrics: AAS=0.962, CUR=0.344, RM=0.031, GFE=0.150

Analyzing: A[0:1] → B[6:]
  Layer A shape: (1280, 320)
  Layer B shape: (256, 1152)
  Overall Match: 46.7%
  Metrics: AAS=0.957, CUR=0.352, RM=0.034, GFE=0.088

Analyzing: A[0:1] → B[7:]
  Layer A shape: (1280, 320)
  Layer B shape: (512, 2304)
  Overall Match: 50.0%
  Metrics: AAS=0.948, CUR=0.529, RM=0.045, GFE=0.072

Analyzing: A[0:1] → B[8:]
  ⚠ No extractable weights, skipping

Analyzing: A[0:2] → B[0:]
  ⚠ No extractable weights, skipping

Analyzing: A[0:2] → B[1:]
  ⚠ No extractable weights, skipping

Analyzing: A[0:2] → B[2:]
  ⚠ No extractable weights, skipping

Analyzing: A[0:2] → B[3:]
  ⚠ No extractable weights, skipping

Analyzing: A[0:2] → B[4:]
  ⚠ No extractable weights, skipping

Analyzing: A[0:2] → B[5:]
  ⚠ No extractable weights, skipping

Analyzing: A[0:2] → B[6:]
  ⚠ No extractable weights, skipping

Analyzing: A[0:2] → B[7:]
  ⚠ No extractable weights, skipping

Analyzing: A[0:2] → B[8:]
  ⚠ No extractable weights, skipping

================================================================================  
BEST CUT POINT FOUND
================================================================================  
Cut A at: 1
Cut B at: 7
Overall Score: 50.0%
Metrics:
  AAS: 0.948
  CUR: 0.529
  RM: 0.045
  GFE: 0.072
  IPI: 0.376
  ASM: 1.000

================================================================================  
GENERATING VISUALIZATIONS
================================================================================  
✓ Visualizations saved
✓ Report saved to hybrid_analysis_results\hybrid_optimization_report.txt

================================================================================  
COMPARING BASELINE VS OPTIMIZED HYBRID
================================================================================  

>>> Creating BASELINE hybrid (cut both in half)...

Hybrid Construction:
  Model A: 3 layers, using first 1
  Model B: 10 layers, using last 6
  Decoder: using 4 convolutional blocks, will add pooling+classifier
  Encoder output: torch.Size([1, 1280, 7, 7])
  Decoder first layer: Conv2d
    Expects Conv2d input: in_channels=64
  Adapter created: Conv2d
  After adapter: torch.Size([1, 64, 7, 7])
  After decoder: torch.Size([1, 512, 1, 1])
  ✓ Pipeline validation successful!
  Added pooling+classifier head: 512→10
HybridModel: adapter created: Conv2d, params=81984

==================================================
Baseline Hybrid Summary
==================================================
Total params: 15,261,638
Trainable params: 15,261,638
Non-trainable params: 0

Encoder: 4,007,548 | Adapter: 81,984 | Decoder: 11,172,106
==================================================
Training baseline adapter...
Epoch 1, Loss: 2.2730
Epoch 2, Loss: 2.0559
Epoch 3, Loss: 1.9357
Epoch 4, Loss: 1.8668
Epoch 5, Loss: 1.8297
Epoch 6, Loss: 1.7906
Epoch 7, Loss: 1.7620
Epoch 8, Loss: 1.7412
Epoch 9, Loss: 1.7211
Epoch 10, Loss: 1.7086
Epoch 11, Loss: 1.6943
Epoch 12, Loss: 1.6736
Epoch 13, Loss: 1.6688
Epoch 14, Loss: 1.6557
Epoch 15, Loss: 1.6448
Baseline Accuracy: 52.73%

>>> Creating OPTIMIZED hybrid (A:1, B:7)...

Hybrid Construction:
  Model A: 3 layers, using first 1
  Model B: 10 layers, using last 3
  Decoder: using 1 convolutional blocks, will add pooling+classifier
  Encoder output: torch.Size([1, 1280, 7, 7])
  Decoder first layer: Conv2d
    Expects Conv2d input: in_channels=256
  Adapter created: Conv2d
  After adapter: torch.Size([1, 256, 7, 7])
  After decoder: torch.Size([1, 512, 4, 4])
  ✓ Pipeline validation successful!
  Added pooling+classifier head: 512→10
HybridModel: adapter created: Conv2d, params=327936

==================================================
Optimized Hybrid Summary
==================================================
Total params: 12,734,342
Trainable params: 12,734,342
Non-trainable params: 0

Encoder: 4,007,548 | Adapter: 327,936 | Decoder: 8,398,858
==================================================
Training optimized adapter...
Epoch 1, Loss: 1.0122
Epoch 2, Loss: 0.7589
Epoch 3, Loss: 0.6887
Epoch 4, Loss: 0.6565
Epoch 5, Loss: 0.6226
Epoch 6, Loss: 0.6059
Epoch 7, Loss: 0.5864
Epoch 8, Loss: 0.5733
Epoch 9, Loss: 0.5600
Epoch 10, Loss: 0.5562
Epoch 11, Loss: 0.5439
Epoch 12, Loss: 0.5340
Epoch 13, Loss: 0.5295
Epoch 14, Loss: 0.5206
Epoch 15, Loss: 0.5147
Optimized Accuracy: 84.27%

================================================================================  
RESULTS COMPARISON
================================================================================  
Baseline Hybrid:  52.73%
Optimized Hybrid: 84.27%
Improvement:      +31.54%

✓ Layer matching analysis successfully improved hybrid performance!

✓ Comparison plot saved to hybrid_analysis_results\hybrid_comparison.png

================================================================================  
ANALYSIS COMPLETE!
================================================================================  
Results saved to: hybrid_analysis_results/
  • cut_point_heatmap.png - Match scores for all configurations
  • best_cut_point_analysis.png - Detailed analysis of optimal cut
  • top_cut_points_comparison.png - Top 5 configurations compared
  • hybrid_comparison.png - Performance comparison
  • hybrid_optimization_report.txt - Detailed text report

```
