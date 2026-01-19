import torch
import torch.nn as nn
import os
from torchvision import transforms, models
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision.models import EfficientNet_B0_Weights, ResNet18_Weights


def _adapt_head(model: nn.Module, num_classes: int = 10) -> nn.Module:
    """Adapt common torchvision model heads to `num_classes` (best-effort)."""
    # EfficientNet-style classifier (Sequential ending with Linear)
    if hasattr(model, 'classifier') and isinstance(model.classifier, (nn.Sequential,)):
        last = model.classifier[-1]
        if isinstance(last, nn.Linear):
            in_f = last.in_features
            model.classifier[-1] = nn.Linear(in_f, num_classes)
        else:
            model.classifier = nn.Sequential(
                nn.Linear(getattr(last, 'in_features', 1280), num_classes))
        return model

    # ResNet-style fc
    if hasattr(model, 'fc') and isinstance(model.fc, nn.Linear):
        in_f = model.fc.in_features
        model.fc = nn.Linear(in_f, num_classes)
        return model

    # Fallback: try to replace last Linear module found
    for name, module in reversed(list(model.named_modules())):
        if isinstance(module, nn.Linear):
            # Don't attempt complex setattr, just leave as-is
            return model

    return model


def load_pretrained(path: str, num_classes: int = 10, device: torch.device | None = None) -> nn.Module:
    """Load a pretrained model, adapt head to `num_classes`, and move to `device`."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if path == 'efficientnet.pth':
        try:
            weights = EfficientNet_B0_Weights.DEFAULT
            model = models.efficientnet_b0(weights=weights)
        except Exception:
            model = models.efficientnet_b0(pretrained=True)
    elif path == 'biasnn.pth':
        try:
            weights = ResNet18_Weights.DEFAULT
            model = models.resnet18(weights=weights)
        except Exception:
            model = models.resnet18(pretrained=True)
    else:
        model = torch.load(path)

    model = _adapt_head(model, num_classes=num_classes)
    model.to(device)
    model.eval()
    model.summary() if hasattr(model, 'summary') else None
    return model


class HybridModel(nn.Module):
    """Hybrid model that stitches parts of model A and model B with an adapter."""

    def __init__(self, model_a_path: str, model_b_path: str, cut_point: int | None = None, 
                 input_shape=(3, 224, 224), device: torch.device | None = None, 
                 cut_by_half: bool = True, cut_a: int | None = None, cut_b: int | None = None):
        super().__init__()
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self._needs_unflatten = False  # Initialize flag
        
        # Load models
        model_a = load_pretrained(model_a_path, device=device)
        model_b = load_pretrained(model_b_path, device=device)
        
        # Determine cut points
        children_a = list(model_a.children())
        children_b = list(model_b.children())
        
        # Use explicit cut points if provided, otherwise determine automatically
        if cut_a is not None and cut_b is not None:
            # Use user-specified cut points
            print(f"\nHybrid Construction:")
            print(f"  Model A: {len(children_a)} layers, using first {cut_a}")
            print(f"  Model B: {len(children_b)} layers, using last {len(children_b) - cut_b}")
        else:
            # Auto-determine cut points
            cut_a, cut_b = self._determine_cut_points(children_a, children_b, cut_point, cut_by_half)
            print(f"\nHybrid Construction:")
            print(f"  Model A: {len(children_a)} layers, using first {cut_a}")
            print(f"  Model B: {len(children_b)} layers, using last {len(children_b) - cut_b}")
        
        # Create encoder/decoder
        self.encoder = nn.Sequential(*children_a[:cut_a]).to(device)
        decoder_layers = children_b[cut_b:]
        
        # For ResNet-style decoders: if we have residual blocks but also pooling+Linear,
        # use only the residual blocks and add our own pooling+classifier
        has_pool_linear = any(isinstance(l, (nn.AdaptiveAvgPool2d, nn.Linear)) for l in decoder_layers)
        if has_pool_linear:
            # Remove pooling and Linear layers, keep only Sequential blocks
            conv_layers = [l for l in decoder_layers if isinstance(l, nn.Sequential)]
            if conv_layers:
                self.decoder = nn.Sequential(*conv_layers).to(device)
                self._decoder_needs_head = True
                print(f"  Decoder: using {len(conv_layers)} convolutional blocks, will add pooling+classifier")
            else:
                self.decoder = nn.Sequential(*decoder_layers).to(device)
                self._decoder_needs_head = False
        else:
            self.decoder = nn.Sequential(*decoder_layers).to(device)
            self._decoder_needs_head = False
        
        # Analyze input/output shapes
        dummy = torch.randn(1, *input_shape, device=device)
        with torch.no_grad():
            enc_out = self.encoder(dummy)
        
        # Extract encoder output info
        enc_info = self._analyze_encoder_output(enc_out)
        print(f"  Encoder output: {enc_out.shape}")
        
        # Get decoder input requirements
        dec_first_layer = self._get_first_meaningful_layer(self.decoder)
        print(f"  Decoder first layer: {type(dec_first_layer).__name__ if dec_first_layer else 'None'}")
        if dec_first_layer:
            if isinstance(dec_first_layer, nn.Conv2d):
                print(f"    Expects Conv2d input: in_channels={dec_first_layer.in_channels}")
            elif isinstance(dec_first_layer, nn.Linear):
                print(f"    Expects Linear input: in_features={dec_first_layer.in_features}")
        
        # Create adapter based on compatibility analysis
        self.adapter = self._create_compatible_adapter(enc_info, dec_first_layer)
        if self.adapter:
            print(f"  Adapter created: {type(self.adapter).__name__}")
        else:
            print(f"  No adapter needed (shapes compatible)")
        
        # Validate the full pipeline
        if not self._validate_pipeline(enc_out):
            # If validation fails, create fallback pipeline
            self._create_fallback_pipeline(enc_info, model_b)
        elif getattr(self, '_decoder_needs_head', False):
            # Add pooling + classifier head for ResNet-style decoders
            num_classes = self._extract_num_classes(model_b)
            # Determine decoder output channels by running a test forward
            with torch.no_grad():
                test = enc_out
                if self.adapter is not None:
                    if isinstance(self.adapter, nn.Conv2d):
                        test = self.adapter(test)
                    else:
                        if test.ndim > 2:
                            test = test.view(test.size(0), -1)
                        test = self.adapter(test)
                dec_out = self.decoder(test)
                out_channels = dec_out.shape[1] if dec_out.ndim == 4 else dec_out.shape[1]
            
            # Append pooling + classifier
            self.decoder = nn.Sequential(
                self.decoder,
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(1),
                nn.Linear(out_channels, num_classes)
            ).to(device)
            print(f"  Added pooling+classifier head: {out_channels}→{num_classes}")
        
        # Debug info
        self._print_adapter_info()
    
    def _determine_cut_points(self, children_a, children_b, cut_point, cut_by_half):
        """Calculate where to cut both models at last convolutional block."""
        # Find last Sequential block with Conv layers (exclude pooling and Linear)
        def find_last_conv_block(children):
            last_seq = -1
            for i, layer in enumerate(children):
                # Only consider Sequential layers with Conv2d
                if isinstance(layer, nn.Sequential):
                    has_conv = any(isinstance(m, nn.Conv2d) for m in layer.modules())
                    if has_conv:
                        last_seq = i
                # Also consider standalone Conv2d layers
                elif isinstance(layer, nn.Conv2d):
                    last_seq = i
            # If found conv blocks, return position after last one
            # Otherwise return safe fallback
            if last_seq >= 0:
                return last_seq + 1
            return max(1, len(children) - 2)
        
        usable_a = find_last_conv_block(children_a)
        usable_b = find_last_conv_block(children_b)
        
        if cut_by_half or cut_point is None:
            # Ensure at least 1 layer in encoder
            cut_a = max(1, usable_a // 2)
            cut_b = max(1, usable_b // 2)
            return cut_a, cut_b
        return max(1, min(int(cut_point), usable_a)), max(1, min(int(cut_point), usable_b))
    
    def _analyze_encoder_output(self, enc_out):
        """Extract information about encoder output shape."""
        enc_shape = enc_out.shape
        enc_flat_dim = enc_out.view(enc_out.size(0), -1).shape[1]
        
        info = {
            'shape': enc_shape,
            'ndim': enc_out.ndim,
            'flat_dim': enc_flat_dim,
            'has_spatial': enc_out.ndim > 2
        }
        
        if info['has_spatial']:
            info.update({
                'channels': enc_shape[1],
                'spatial_dims': tuple(enc_shape[2:])
            })
        
        return info
    
    def _get_first_meaningful_layer(self, module):
        """Find first Conv2d or Linear layer in decoder."""
        for m in module.modules():
            if m is module:
                continue
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                return m
        return None
    
    def _create_compatible_adapter(self, enc_info, dec_first_layer):
        """Create appropriate adapter based on encoder output and decoder input."""
        # Case 1: Encoder has spatial dimensions
        if enc_info['has_spatial']:
            return self._create_spatial_adapter(enc_info, dec_first_layer)
        # Case 2: Encoder is flattened
        else:
            return self._create_flattened_adapter(enc_info, dec_first_layer)
    
    def _create_spatial_adapter(self, enc_info, dec_first_layer):
        """Create adapter for spatial (4D) encoder output."""
        enc_ch = enc_info['channels']
        
        if isinstance(dec_first_layer, nn.Conv2d):
            dec_in_ch = dec_first_layer.in_channels
            if enc_ch != dec_in_ch:
                return nn.Conv2d(enc_ch, dec_in_ch, kernel_size=1).to(self.device)
        
        elif isinstance(dec_first_layer, nn.Linear):
            if enc_info['flat_dim'] != dec_first_layer.in_features:
                return nn.Linear(enc_info['flat_dim'], dec_first_layer.in_features).to(self.device)
        
        return None
    
    def _create_flattened_adapter(self, enc_info, dec_first_layer):
        """Create adapter for flattened (2D) encoder output."""
        enc_dim = enc_info['flat_dim']
        
        if isinstance(dec_first_layer, nn.Conv2d):
            # Need to map flattened to channels for Conv2d
            dec_in_ch = dec_first_layer.in_channels
            adapter = nn.Linear(enc_dim, dec_in_ch).to(self.device)
            self._needs_unflatten = True
            return adapter
        
        elif isinstance(dec_first_layer, nn.Linear):
            if enc_dim != dec_first_layer.in_features:
                return nn.Linear(enc_dim, dec_first_layer.in_features).to(self.device)
        
        # Default: small MLP adapter
        hidden = max(64, min(512, enc_dim // 4))
        return nn.Sequential(
            nn.Linear(enc_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
        ).to(self.device)
    
    def _validate_pipeline(self, enc_out):
        """Test if encoder -> adapter -> decoder works."""
        try:
            with torch.no_grad():
                test = enc_out
                if self.adapter is not None:
                    # Apply adapter inline for validation
                    if isinstance(self.adapter, nn.Conv2d):
                        test = self.adapter(test)
                    else:
                        if test.ndim > 2:
                            test = test.view(test.size(0), -1)
                        test = self.adapter(test)
                        if self._needs_unflatten:
                            test = test.unsqueeze(-1).unsqueeze(-1)
                print(f"  After adapter: {test.shape}")
                result = self.decoder(test)
                print(f"  After decoder: {result.shape}")
                print(f"  ✓ Pipeline validation successful!")
            return True
        except Exception as e:
            print(f"  ✗ Pipeline validation failed: {e}")
            return False
    
    def _apply_adapter_forward(self, x):
        """Apply adapter during forward pass (for validation)."""
        if isinstance(self.adapter, nn.Conv2d):
            return self.adapter(x)
        else:
            # Linear adapter
            if x.ndim > 2:
                x = x.view(x.size(0), -1)
            x = self.adapter(x)
            if getattr(self, '_needs_unflatten', False):
                x = x.unsqueeze(-1).unsqueeze(-1)
            return x
    
    def _create_fallback_pipeline(self, enc_info, model_b):
        """Create simple fallback pipeline when stitching fails."""
        print('Creating fallback pipeline...')
        
        # Get number of classes
        num_classes = self._extract_num_classes(model_b)
        enc_dim = enc_info['flat_dim']
        
        # Two-stage reduction for very large encoder outputs
        if enc_dim > 10000:
            hidden1 = 512
            hidden2 = 128
            self.adapter = nn.Sequential(
                nn.Flatten(1),
                nn.Linear(enc_dim, hidden1),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(hidden1, hidden2),
                nn.ReLU(inplace=True)
            ).to(self.device)
            adapter_out = hidden2
        else:
            # Single-stage for smaller encoders
            hidden = min(256, enc_dim // 4)
            self.adapter = nn.Sequential(
                nn.Flatten(1),
                nn.Linear(enc_dim, hidden),
                nn.ReLU(inplace=True)
            ).to(self.device)
            adapter_out = hidden
        
        # Replace decoder with simple classifier
        self.decoder = nn.Sequential(
            nn.Linear(adapter_out, num_classes)
        ).to(self.device)
    
    def _extract_num_classes(self, model):
        """Extract number of classes from model's head."""
        if hasattr(model, 'fc') and isinstance(model.fc, nn.Linear):
            return model.fc.out_features
        elif hasattr(model, 'classifier') and isinstance(model.classifier, nn.Sequential):
            last = model.classifier[-1]
            if isinstance(last, nn.Linear):
                return last.out_features
        return 10  # Default for CIFAR-10
    
    def _print_adapter_info(self):
        """Print debug information about adapter."""
        if self.adapter is not None:
            # Ensure adapter parameters are trainable
            for p in self.adapter.parameters():
                p.requires_grad = True
            param_count = sum(p.numel() for p in self.adapter.parameters())
            print(f'HybridModel: adapter created: {type(self.adapter).__name__}, params={param_count}')
        else:
            print('HybridModel: no adapter created')
    
    def forward(self, x):
        """Forward pass through encoder -> adapter -> decoder."""
        x = x.to(self.device)
        x = self.encoder(x)
        
        if self.adapter is not None:
            x = self._apply_adapter_forward(x)
        
        x = self.decoder(x)
        return x


def freeze(model: nn.Module):
    """Freeze all parameters in a model."""
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_all(model: nn.Module):
    """Unfreeze all parameters in a model."""
    for param in model.parameters():
        param.requires_grad = True


def train_only_adapter(model: HybridModel):
    """Freeze encoder and decoder, only train adapter."""
    freeze(model.encoder)
    freeze(model.decoder)
    
    if model.adapter is not None:
        for p in model.adapter.parameters():
            p.requires_grad = True
    else:
        print('Warning: No adapter found to train')


def train_entire_model(model: nn.Module):
    """Unfreeze all parameters in a model."""
    unfreeze_all(model)


def enable_head(model: nn.Module) -> int:
    """Enable (set requires_grad=True) for a model's classifier head parameters."""
    enabled = 0
    try:
        if hasattr(model, 'classifier') and isinstance(model.classifier, (nn.Sequential,)):
            last = model.classifier[-1]
            if isinstance(last, nn.Linear):
                for p in last.parameters():
                    p.requires_grad = True
                enabled += sum(p.numel() for p in last.parameters())
        if hasattr(model, 'fc') and isinstance(model.fc, nn.Linear):
            for p in model.fc.parameters():
                p.requires_grad = True
            enabled += sum(p.numel() for p in model.fc.parameters())
        if enabled == 0:
            # fallback: find last Linear
            for name, module in reversed(list(model.named_modules())):
                if isinstance(module, nn.Linear):
                    for p in module.parameters():
                        p.requires_grad = True
                    enabled += sum(p.numel() for p in module.parameters())
                    break
    except Exception:
        pass
    return enabled


def train_head(model: nn.Module, epochs: int = 5, dataloader: DataLoader | None = None, 
               device: torch.device | None = None, lr: float = 1e-3):
    """Freeze all parameters and train only the model head."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Freeze everything
    freeze(model)
    
    # Enable head params
    num_enabled = enable_head(model)
    if num_enabled == 0:
        print('train_head: no head parameters found to train.')
        return
    
    # Use provided dataloader or default
    if dataloader is None:
        dataloader = get_cifar10_loader(train=True, download=True, batch_size=64)
    
    # Build optimizer only for head params
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    model.to(device)
    model.train()
    
    history = {'loss': []}
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        avg_loss = running_loss / len(dataloader)
        history['loss'].append(avg_loss)
        print(f"Head Epoch {epoch+1}, Loss: {avg_loss:.4f}")
    
    # After training, set model to eval mode
    model.eval()
    return history


def make_default_transform(resize: int = 224):
    """Create default transform with ImageNet normalization."""
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])


def get_cifar10_loader(root: str = './data', train: bool = False, batch_size: int = 64, 
                       shuffle: bool = True, download: bool = True, transform=None):
    """Create DataLoader for CIFAR-10."""
    if transform is None:
        transform = make_default_transform()
    ds = CIFAR10(root=root, train=train, download=download, transform=transform)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device | None = None) -> float:
    """Evaluate model accuracy on a dataloader."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.to(device)
    model.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            if outputs.ndim == 1:
                continue  # single-value output: skip
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return correct / total if total > 0 else 0.0


def train_model(model: nn.Module, epochs: int = 5, dataloader: DataLoader | None = None, 
                device: torch.device | None = None, lr: float = 0.001):
    """Generic training function for any model. Returns training history."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.to(device)
    
    if dataloader is None:
        dataloader = get_cifar10_loader(train=True, download=True, batch_size=64)
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        print('No trainable parameters found. Skipping training.')
        return {'loss': []}
    
    optimizer = torch.optim.Adam(trainable_params, lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    
    history = {'loss': []}
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        avg_loss = running_loss / len(dataloader)
        history['loss'].append(avg_loss)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
    
    model.eval()
    return history


def collect_activation_stats(model: nn.Module, sample_batch: torch.Tensor, 
                            device: torch.device | None = None, max_modules: int | None = 200):
    """Collect mean absolute activation per module for a single forward pass."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    model.eval()
    
    stats: dict[str, float] = {}
    handles = []
    
    def make_hook(name):
        def hook(module, inp, out):
            try:
                a = out.detach()
                stats[name] = float(a.abs().mean().cpu().item())
            except Exception:
                stats[name] = float(0.0)
        return hook
    
    # Register hooks on Conv2d and Linear modules
    count = 0
    for name, module in model.named_modules():
        if module is model:
            continue
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            handles.append(module.register_forward_hook(make_hook(name)))
            count += 1
            if max_modules is not None and count >= max_modules:
                break
    
    # Run forward with a single batch
    with torch.no_grad():
        sample = sample_batch.to(device)
        try:
            _ = model(sample)
        except Exception:
            # Try flattening per-sample
            try:
                s = sample.view(sample.size(0), -1).to(device)
                _ = model(s)
            except Exception:
                pass
    
    # Remove hooks
    for h in handles:
        h.remove()
    
    return stats


def plot_activation_stats(stats: dict[str, float], title: str, out_path: str, top_k: int | None = 50):
    """Plot a bar chart of the top_k modules by mean-abs activation and save PNG."""
    try:
        import matplotlib.pyplot as plt
    except Exception:
        # matplotlib not available; write CSV fallback
        import csv
        dirpath = os.path.dirname(out_path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        with open(out_path.replace('.png', '.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['module', 'mean_abs_activation'])
            for k, v in stats.items():
                w.writerow([k, v])
        return
    
    items = sorted(stats.items(), key=lambda kv: kv[1], reverse=True)
    if top_k is not None:
        items = items[:top_k]
    
    names = [k for k, _ in items]
    values = [v for _, v in items]
    
    dirpath = os.path.dirname(out_path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    
    plt.figure(figsize=(max(6, len(names) * 0.25), 6))
    plt.barh(range(len(names))[::-1], values, align='center')
    plt.yticks(range(len(names))[::-1], names, fontsize=8)
    plt.xlabel('Mean abs activation')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def get_sample_batch(batch_size: int = 8, resize: int = 224):
    """Return a single batch from CIFAR-10 test split using default transform."""
    loader = get_cifar10_loader(train=False, download=True, batch_size=batch_size)
    for xb, yb in loader:
        return xb
    raise RuntimeError('Could not load sample batch')