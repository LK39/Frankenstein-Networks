import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from model import (HybridModel, train_model, train_only_adapter, evaluate, 
                   load_pretrained, get_cifar10_loader, train_head,
                   collect_activation_stats, plot_activation_stats, get_sample_batch)

CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def plot_training_curves(history, title, filepath):
    """Plot training loss and accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs = range(1, len(history['loss']) + 1)
    ax1.plot(epochs, history['loss'], 'b-', linewidth=2)
    ax1.set_title(f'{title} - Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)
    
    if 'accuracy' in history:
        ax2.plot(epochs, history['accuracy'], 'g-', linewidth=2)
        ax2.set_title(f'{title} - Training Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

def plot_predictions(model, title, filepath, num_samples=16):
    """Visualize model predictions on sample images."""
    test_loader = get_cifar10_loader(train=False, batch_size=num_samples, shuffle=True)
    images, labels = next(iter(test_loader))
    
    device = next(model.parameters()).device
    images = images.to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
    
    # Un Normalize images for display
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    images_cpu = images.cpu() * std + mean
    images_cpu = images_cpu.clamp(0, 1)
    
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for idx, ax in enumerate(axes.flat):
        if idx >= num_samples:
            break
        img = images_cpu[idx].permute(1, 2, 0).numpy()
        ax.imshow(img)
        
        pred_label = CIFAR10_CLASSES[preds[idx]]
        true_label = CIFAR10_CLASSES[labels[idx]]
        color = 'green' if preds[idx] == labels[idx] else 'red'
        ax.set_title(f'P: {pred_label}\nT: {true_label}', color=color, fontsize=9)
        ax.axis('off')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

def print_summary(model, name="Model"):
    """Print model architecture summary with parameter counts."""
    print(f"\n{'='*50}\n{name} Summary\n{'='*50}")
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total:,}")
    print(f"Trainable params: {trainable:,}")
    print(f"Non-trainable params: {total - trainable:,}")
    if hasattr(model, 'encoder'):
        enc = sum(p.numel() for p in model.encoder.parameters())
        dec = sum(p.numel() for p in model.decoder.parameters())
        adp = sum(p.numel() for p in model.adapter.parameters()) if model.adapter else 0
        print(f"\nEncoder: {enc:,} | Adapter: {adp:,} | Decoder: {dec:,}")
    print('='*50)

def print_architecture(model, name="Model"):
    """Print detailed layer-by-layer architecture breakdown."""
    print(f"\n{name} - Architecture Breakdown:")
    print("-" * 70)
    children = list(model.children())
    for i, layer in enumerate(children):
        params = sum(p.numel() for p in layer.parameters())
        print(f"  [{i}] {type(layer).__name__:30s} {params:>12,} params")
    print("-" * 70)

def analyze_layer_outputs(model, name="Model"):
    """Analyze output shapes at each layer to help determine cut points."""
    print(f"\n{name} - Layer Output Shapes:")
    print("-" * 60)
    dummy = get_sample_batch(batch_size=1)
    if next(model.parameters()).is_cuda:
        dummy = dummy.cuda()
    
    x = dummy
    children = list(model.children())
    for i, layer in enumerate(children):
        try:
            # Auto-flatten if next layer is Linear and current output is spatial
            if x.ndim > 2 and isinstance(layer, (nn.Sequential,)):
                # Check if Sequential contains Linear
                has_linear = any(isinstance(m, nn.Linear) for m in layer.modules())
                if has_linear:
                    x = x.flatten(1)
            x = layer(x)
            params = sum(p.numel() for p in layer.parameters())
            print(f"  [{i}] {type(layer).__name__:20s} → {str(tuple(x.shape)):20s} ({params:,} params)")
        except Exception as e:
            print(f"  [{i}] {type(layer).__name__:20s} → Error: {e}")
            break
    print("-" * 60)

if __name__ == "__main__":
    os.makedirs('outputs', exist_ok=True)
    
    # Setup
    train_loader = get_cifar10_loader(train=True, download=True, batch_size=64)
    test_loader = get_cifar10_loader(train=False, download=True, batch_size=128)
    
    # Load and train EfficientNet head
    print("\n>>> EfficientNet-B0")
    eff = load_pretrained('efficientnet.pth')
    print_summary(eff, "EfficientNet-B0")
    print_architecture(eff, "EfficientNet-B0")
    history_eff = train_head(eff, epochs=15, dataloader=train_loader)
    acc_eff = evaluate(eff, test_loader)
    print(f"Accuracy: {acc_eff:.4f}")
    plot_training_curves(history_eff, 'EfficientNet-B0', 'outputs/eff_training.png')
    plot_predictions(eff, 'EfficientNet-B0 Predictions', 'outputs/eff_predictions.png')
    
    # Load and train BIASNN head
    print("\n>>> BIASNN (ResNet-18)")
    bias = load_pretrained('biasnn.pth')
    print_summary(bias, "BIASNN/ResNet-18")
    print_architecture(bias, "BIASNN/ResNet-18")
    history_bias = train_head(bias, epochs=15, dataloader=train_loader)
    acc_bias = evaluate(bias, test_loader)
    print(f"Accuracy: {acc_bias:.4f}")
    plot_training_curves(history_bias, 'ResNet-18', 'outputs/bias_training.png')
    plot_predictions(bias, 'ResNet-18 Predictions', 'outputs/bias_predictions.png')
    
    # Visualize activations
    print("\n>>> Creating Activation Visualizations")
    sample = get_sample_batch(batch_size=8)
    stats_eff = collect_activation_stats(eff, sample)
    stats_bias = collect_activation_stats(bias, sample)
    plot_activation_stats(stats_eff, 'EfficientNet-B0 Activations', 'outputs/eff_activations.png')
    plot_activation_stats(stats_bias, 'ResNet-18 Activations', 'outputs/bias_activations.png')
    print("✓ Saved activation plots to outputs/")
    
    # Create and train hybrid
    print("\n>>> Hybrid Model")
    print("Using EfficientNet features 0-4 (richer mid-level features) + ResNet layers 4-7 (residual decoder)")
    # EfficientNet has 3 top-level layers: [0]=features, [1]=pool, [2]=classifier
    # To use features 0-4, we need to extract from within the features Sequential
    # Instead, use cut_a=1 (just features block) and cut_b=6 (ResNet layers 4-7)
    hybrid = HybridModel('efficientnet.pth', 'biasnn.pth', cut_a=1, cut_b=6)
    print_summary(hybrid, "Hybrid Model")
    
    # Show what layers are actually in encoder/decoder
    print("\nHybrid Architecture Breakdown:")
    print("Encoder layers:")
    for i, layer in enumerate(hybrid.encoder):
        params = sum(p.numel() for p in layer.parameters())
        print(f"  [{i}] {type(layer).__name__:30s} {params:>12,} params")
    print("\nDecoder layers:")
    for i, layer in enumerate(hybrid.decoder):
        params = sum(p.numel() for p in layer.parameters())
        print(f"  [{i}] {type(layer).__name__:30s} {params:>12,} params")
    if hybrid.adapter:
        print(f"\nAdapter: {type(hybrid.adapter).__name__}")
        if isinstance(hybrid.adapter, nn.Sequential):
            for i, layer in enumerate(hybrid.adapter):
                params = sum(p.numel() for p in layer.parameters())
                print(f"  [{i}] {type(layer).__name__:30s} {params:>12,} params")
    
    # Train adapter only (freeze encoder+decoder)
    print("\n>>> Training Adapter Only (15 epochs, lr=1e-3)")
    train_only_adapter(hybrid)
    history_hybrid = train_model(hybrid, epochs=15, dataloader=train_loader, lr=1e-3)
    acc_hybrid = evaluate(hybrid, test_loader)
    print(f"Adapter-Only Accuracy: {acc_hybrid:.4f}")
    plot_training_curves(history_hybrid, 'Hybrid Model - Adapter Only', 'outputs/hybrid_training.png')
    plot_predictions(hybrid, 'Hybrid Model Predictions', 'outputs/hybrid_predictions.png')
    
    # Hybrid activations
    stats_hybrid = collect_activation_stats(hybrid, sample)
    plot_activation_stats(stats_hybrid, 'Hybrid Model Activations', 'outputs/hybrid_activations.png')
    
    print(f"\n{'='*50}")
    print(f"Final Results: Eff={acc_eff:.4f} | Bias={acc_bias:.4f} | Hybrid={acc_hybrid:.4f}")
    print(f"{'='*50}")