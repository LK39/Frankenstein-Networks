"""
Analyze CNN models using Layer Matching Framework and build optimized hybrid.

This script:
1. Analyzes EfficientNet-B0 and ResNet-18 layer-by-layer
2. Identifies optimal cut points based on matching metrics
3. Creates an optimized hybrid model
4. Compares performance against baseline hybrid
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from model import load_pretrained, HybridModel, train_model, get_cifar10_loader, train_only_adapter
from main import evaluate, print_summary
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
OUTPUT_DIR = "hybrid_analysis_results"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# ============================================================================


class CNNLayerMatchAnalyzer:
    """Analyzes CNN models and finds optimal layer matching for hybrid construction."""
    
    def __init__(self, model_a_path, model_b_path, device=None):
        self.model_a_path = model_a_path
        self.model_b_path = model_b_path
        self.device = device or DEVICE
        
        # Load models
        print("Loading models...")
        self.model_a = load_pretrained(model_a_path, device=self.device)
        self.model_b = load_pretrained(model_b_path, device=self.device)
        
        # Extract layers
        self.layers_a = list(self.model_a.children())
        self.layers_b = list(self.model_b.children())
        
        print(f"✓ Model A ({model_a_path}): {len(self.layers_a)} top-level layers")
        print(f"✓ Model B ({model_b_path}): {len(self.layers_b)} top-level layers")
        
        self.analysis_results = []
        self.best_cut_point = None
        
    def extract_layer_weights(self, layer):
        """Extract weight matrices from a layer or sequential."""
        weights = []
        
        if isinstance(layer, nn.Sequential):
            for sublayer in layer:
                weights.extend(self.extract_layer_weights(sublayer))
        elif isinstance(layer, (nn.Conv2d, nn.Linear)):
            w = layer.weight.detach().cpu().numpy()
            # Reshape Conv2d to 2D: (out_channels, in_channels * kernel_h * kernel_w)
            if len(w.shape) == 4:
                w = w.reshape(w.shape[0], -1)
            weights.append({
                'type': type(layer).__name__,
                'weight': w,
                'shape': w.shape
            })
        elif hasattr(layer, 'children'):
            for child in layer.children():
                weights.extend(self.extract_layer_weights(child))
                
        return weights
    
    def calculate_layer_metrics(self, W_A, W_B, activation=None):
        """Calculate matching metrics between two weight matrices."""
        # Simulate activation if not provided
        if activation is None:
            activation = np.random.randn(W_A.shape[0])
            activation = np.maximum(0, activation)  # ReLU
            activation = activation / (np.linalg.norm(activation) + 1e-8)
        
        # Ensure compatible dimensions
        if W_A.shape[0] != W_B.shape[1]:
            # Create compatible activation
            activation = np.random.randn(W_B.shape[1])
            activation = np.maximum(0, activation)
            activation = activation / (np.linalg.norm(activation) + 1e-8)
        
        # 1. Activation Alignment Score (AAS)
        projected = W_B @ activation
        projection_norm = np.linalg.norm(projected)
        W_B_norm = np.linalg.norm(W_B, 'fro')
        aas = max(0, 1 - (projection_norm / (W_B_norm + 1e-8)))
        
        # 2. Capacity Utilization Ratio (CUR)
        neuron_activations = np.abs(projected)
        threshold = 0.3 * np.max(neuron_activations)
        cur = np.sum(neuron_activations > threshold) / len(neuron_activations)
        
        # 3. Representational Match (RM)
        cosine_sims = []
        for weight_vec in W_B:
            norm_prod = np.linalg.norm(weight_vec) * np.linalg.norm(activation)
            if norm_prod > 0:
                cos_sim = np.abs(np.dot(weight_vec, activation) / norm_prod)
                cosine_sims.append(cos_sim)
        rm = np.mean(cosine_sims) if cosine_sims else 0
        
        # 4. Gradient Flow Efficiency (GFE)
        try:
            U, S, Vt = np.linalg.svd(W_B, full_matrices=False)
            if len(S) > 0 and S[0] > 0:
                gfe = S[-1] / S[0]
            else:
                gfe = 0
            sv = S
        except:
            gfe = 0.5
            sv = np.ones(min(W_B.shape))
        
        # 5. Information Preservation Index (IPI)
        normalized_sv = sv / (np.sum(sv) + 1e-8)
        entropy = -np.sum(normalized_sv * np.log(normalized_sv + 1e-8))
        ipi = np.exp(-entropy / (np.log(len(normalized_sv)) + 1e-8))
        
        # 6. Activation Sparsity Match (ASM)
        act_sparsity = np.sum(np.abs(activation) > 0.1) / len(activation)
        weight_sparsity = np.sum(np.abs(W_B) > 0.2) / W_B.size
        asm = 1 - abs(act_sparsity - weight_sparsity)
        
        # Overall score
        weights = [0.25, 0.20, 0.20, 0.15, 0.10, 0.10]
        metrics_values = [aas, cur, rm, gfe, ipi, asm]
        overall = sum(w * m for w, m in zip(weights, metrics_values))
        
        return {
            'aas': aas, 'cur': cur, 'rm': rm,
            'gfe': gfe, 'ipi': ipi, 'asm': asm,
            'overall': overall,
            'singular_values': sv,
            'neuron_activations': neuron_activations
        }
    
    def analyze_all_cut_points(self):
        """Analyze all possible cut points between model A and B."""
        print("\n" + "="*80)
        print("ANALYZING ALL CUT POINTS")
        print("="*80)
        
        # We analyze connections between encoder (Model A) and decoder (Model B)
        # For each cut point in A, analyze compatibility with each start point in B
        
        for cut_a in range(1, len(self.layers_a)):
            for cut_b in range(0, len(self.layers_b) - 1):
                print(f"\nAnalyzing: A[0:{cut_a}] → B[{cut_b}:]")
                
                # Get last layer of encoder (Model A)
                encoder_last = self.layers_a[cut_a - 1]
                weights_a = self.extract_layer_weights(encoder_last)
                
                # Get first layer of decoder (Model B)
                decoder_first = self.layers_b[cut_b]
                weights_b = self.extract_layer_weights(decoder_first)
                
                if not weights_a or not weights_b:
                    print("  ⚠ No extractable weights, skipping")
                    continue
                
                # Analyze last weight of A with first weight of B
                W_A = weights_a[-1]['weight']
                W_B = weights_b[0]['weight']
                
                print(f"  Layer A shape: {W_A.shape}")
                print(f"  Layer B shape: {W_B.shape}")
                
                try:
                    metrics = self.calculate_layer_metrics(W_A, W_B)
                    
                    result = {
                        'cut_a': cut_a,
                        'cut_b': cut_b,
                        'encoder_shape': W_A.shape,
                        'decoder_shape': W_B.shape,
                        'metrics': metrics
                    }
                    
                    self.analysis_results.append(result)
                    
                    print(f"  Overall Match: {metrics['overall']*100:.1f}%")
                    print(f"  Metrics: AAS={metrics['aas']:.3f}, CUR={metrics['cur']:.3f}, "
                          f"RM={metrics['rm']:.3f}, GFE={metrics['gfe']:.3f}")
                    
                except Exception as e:
                    print(f"  ✗ Error: {e}")
                    continue
        
        # Find best cut point
        if self.analysis_results:
            self.analysis_results.sort(key=lambda x: x['metrics']['overall'], reverse=True)
            self.best_cut_point = self.analysis_results[0]
            
            print("\n" + "="*80)
            print("BEST CUT POINT FOUND")
            print("="*80)
            print(f"Cut A at: {self.best_cut_point['cut_a']}")
            print(f"Cut B at: {self.best_cut_point['cut_b']}")
            print(f"Overall Score: {self.best_cut_point['metrics']['overall']*100:.1f}%")
            print(f"Metrics:")
            for metric in ['aas', 'cur', 'rm', 'gfe', 'ipi', 'asm']:
                print(f"  {metric.upper()}: {self.best_cut_point['metrics'][metric]:.3f}")
    
    def visualize_results(self, output_dir):
        """Create visualizations of the analysis."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        print(f"\n{'='*80}")
        print("GENERATING VISUALIZATIONS")
        print("="*80)
        
        # 1. Cut point heatmap
        self._plot_cut_point_heatmap(output_path)
        
        # 2. Best cut point detailed analysis
        self._plot_best_cut_point_analysis(output_path)
        
        # 3. Metrics comparison
        self._plot_metrics_comparison(output_path)
        
        print("✓ Visualizations saved")
    
    def _plot_cut_point_heatmap(self, output_path):
        """Plot heatmap showing match scores for all cut point combinations."""
        if not self.analysis_results:
            return
        
        # Create matrix
        max_a = max(r['cut_a'] for r in self.analysis_results) + 1
        max_b = max(r['cut_b'] for r in self.analysis_results) + 1
        
        heatmap_data = np.zeros((max_b, max_a))
        
        for result in self.analysis_results:
            heatmap_data[result['cut_b'], result['cut_a']] = result['metrics']['overall']
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        
        ax.set_xlabel('Model A Cut Point (layers to keep)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Model B Start Point (layers to skip)', fontsize=12, fontweight='bold')
        ax.set_title('Layer Matching Scores: All Cut Point Combinations', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, label='Overall Match Score')
        
        # Mark best point
        if self.best_cut_point:
            ax.plot(self.best_cut_point['cut_a'], self.best_cut_point['cut_b'], 
                   'w*', markersize=20, markeredgecolor='black', markeredgewidth=2,
                   label=f"Best: ({self.best_cut_point['cut_a']}, {self.best_cut_point['cut_b']})")
            ax.legend(fontsize=10)
        
        # Add grid
        ax.set_xticks(range(max_a))
        ax.set_yticks(range(max_b))
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'cut_point_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_best_cut_point_analysis(self, output_path):
        """Detailed analysis of the best cut point."""
        if not self.best_cut_point:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Best Cut Point Analysis: A[:{self.best_cut_point["cut_a"]}] → B[{self.best_cut_point["cut_b"]}:]',
                    fontsize=14, fontweight='bold')
        
        metrics = self.best_cut_point['metrics']
        
        # 1. Metrics radar
        ax = axes[0, 0]
        categories = ['AAS', 'CUR', 'RM', 'GFE', 'IPI', 'ASM']
        values = [metrics['aas'], metrics['cur'], metrics['rm'],
                 metrics['gfe'], metrics['ipi'], metrics['asm']]
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        values_plot = values + values[:1]
        angles_plot = angles + angles[:1]
        
        ax_polar = plt.subplot(2, 2, 1, projection='polar')
        ax_polar.plot(angles_plot, values_plot, 'o-', linewidth=2, color='steelblue')
        ax_polar.fill(angles_plot, values_plot, alpha=0.25, color='steelblue')
        ax_polar.set_xticks(angles)
        ax_polar.set_xticklabels(categories)
        ax_polar.set_ylim(0, 1)
        ax_polar.set_title('Metrics Radar', fontweight='bold', pad=20)
        ax_polar.grid(True)
        
        # 2. Singular values
        ax = axes[0, 1]
        sv = metrics['singular_values'][:20]
        ax.semilogy(sv, 'o-', color='steelblue', linewidth=2, markersize=6, label='Actual')
        ax.semilogy(np.exp(-np.arange(len(sv))/5), '--', color='gray', 
                   alpha=0.7, label='Ideal decay')
        ax.set_xlabel('Singular Value Index')
        ax.set_ylabel('Magnitude (log scale)')
        ax.set_title('Singular Value Spectrum', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # 3. Neuron activations
        ax = axes[1, 0]
        activations = np.sort(metrics['neuron_activations'])
        threshold = 0.3 * np.max(activations)
        colors = ['green' if a >= threshold else 'red' for a in activations]
        ax.bar(range(len(activations)), activations, color=colors, alpha=0.6, width=1.0)
        ax.axhline(y=threshold, color='orange', linestyle='--', 
                  label=f'CUR: {metrics["cur"]*100:.1f}%')
        ax.set_xlabel('Neuron Index (sorted)')
        ax.set_ylabel('Activation Level')
        ax.set_title('Neuron Activation Distribution', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # 4. Metrics bar chart
        ax = axes[1, 1]
        bars = ax.bar(categories, values, color='steelblue', alpha=0.7, edgecolor='navy')
        ax.set_ylabel('Score')
        ax.set_title('Individual Metrics', fontweight='bold')
        ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(output_path / 'best_cut_point_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_metrics_comparison(self, output_path):
        """Compare metrics across top cut points."""
        if len(self.analysis_results) < 5:
            return
        
        # Get top 5 cut points
        top_5 = self.analysis_results[:5]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(top_5))
        width = 0.12
        
        metrics_names = ['aas', 'cur', 'rm', 'gfe', 'ipi', 'asm']
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
        
        for i, metric in enumerate(metrics_names):
            values = [r['metrics'][metric] for r in top_5]
            ax.bar(x + i * width, values, width, label=metric.upper(), 
                  color=colors[i], alpha=0.8)
        
        ax.set_xlabel('Cut Point Configuration', fontweight='bold')
        ax.set_ylabel('Metric Score', fontweight='bold')
        ax.set_title('Top 5 Cut Points: Metrics Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * 2.5)
        ax.set_xticklabels([f"A:{r['cut_a']},B:{r['cut_b']}\n({r['metrics']['overall']*100:.1f}%)" 
                           for r in top_5])
        ax.legend(ncol=6, loc='upper center', bbox_to_anchor=(0.5, -0.1))
        ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'top_cut_points_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_report(self, output_dir):
        """Generate text report of findings."""
        output_path = Path(output_dir)
        report_file = output_path / 'hybrid_optimization_report.txt'
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("HYBRID MODEL OPTIMIZATION REPORT\n")
            f.write("Layer Matching Analysis for CNN Model Stitching\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Model A: {self.model_a_path}\n")
            f.write(f"Model B: {self.model_b_path}\n")
            f.write(f"Total Configurations Analyzed: {len(self.analysis_results)}\n\n")
            
            f.write("-"*80 + "\n")
            f.write("OPTIMAL CUT POINT\n")
            f.write("-"*80 + "\n\n")
            
            if self.best_cut_point:
                f.write(f"Model A: Keep first {self.best_cut_point['cut_a']} layers\n")
                f.write(f"Model B: Use layers from {self.best_cut_point['cut_b']} onwards\n")
                f.write(f"Overall Match Score: {self.best_cut_point['metrics']['overall']*100:.2f}%\n\n")
                
                f.write("Detailed Metrics:\n")
                for metric in ['aas', 'cur', 'rm', 'gfe', 'ipi', 'asm']:
                    value = self.best_cut_point['metrics'][metric]
                    f.write(f"  {metric.upper()}: {value:.4f}\n")
                
                f.write(f"\nEncoder Output Shape: {self.best_cut_point['encoder_shape']}\n")
                f.write(f"Decoder Input Shape: {self.best_cut_point['decoder_shape']}\n")
            
            f.write("\n" + "-"*80 + "\n")
            f.write("TOP 5 CONFIGURATIONS\n")
            f.write("-"*80 + "\n\n")
            
            for i, result in enumerate(self.analysis_results[:5], 1):
                f.write(f"[{i}] A:{result['cut_a']}, B:{result['cut_b']} | "
                       f"Score: {result['metrics']['overall']*100:.1f}%\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("RECOMMENDATIONS\n")
            f.write("="*80 + "\n\n")
            
            if self.best_cut_point:
                score = self.best_cut_point['metrics']['overall']
                if score >= 0.75:
                    f.write("✓ EXCELLENT match found! This hybrid should perform very well.\n")
                    f.write("  Proceed with this configuration for training.\n")
                elif score >= 0.60:
                    f.write("✓ GOOD match found. This hybrid should be viable.\n")
                    f.write("  Consider adding batch normalization in adapter for stability.\n")
                else:
                    f.write("⚠ MODERATE match. Consider:\n")
                    f.write("  1. Adding intermediate adapter layers\n")
                    f.write("  2. Using residual connections\n")
                    f.write("  3. Trying different cut points from top 5 list\n")
        
        print(f"✓ Report saved to {report_file}")


def compare_hybrids(analyzer, output_dir):
    """Compare baseline hybrid vs optimized hybrid."""
    print("\n" + "="*80)
    print("COMPARING BASELINE VS OPTIMIZED HYBRID")
    print("="*80)
    
    # Load data
    test_loader = get_cifar10_loader(train=False, batch_size=128)
    
    # 1. Baseline hybrid (cut both in half)
    print("\n>>> Creating BASELINE hybrid (cut both in half)...")
    baseline_hybrid = HybridModel(
        analyzer.model_a_path,
        analyzer.model_b_path,
        cut_by_half=True
    )
    print_summary(baseline_hybrid, "Baseline Hybrid")
    
    # Train adapter
    print("Training baseline adapter...")
    train_only_adapter(baseline_hybrid)
    train_loader = get_cifar10_loader(train=True, batch_size=64)
    train_model(baseline_hybrid, epochs=15, dataloader=train_loader)
    
    # Evaluate
    baseline_acc = evaluate(baseline_hybrid, test_loader)
    print(f"Baseline Accuracy: {baseline_acc*100:.2f}%")
    
    # 2. Optimized hybrid (using best cut point)
    if analyzer.best_cut_point:
        print(f"\n>>> Creating OPTIMIZED hybrid (A:{analyzer.best_cut_point['cut_a']}, "
              f"B:{analyzer.best_cut_point['cut_b']})...")
        optimized_hybrid = HybridModel(
            analyzer.model_a_path,
            analyzer.model_b_path,
            cut_a=analyzer.best_cut_point['cut_a'],
            cut_b=analyzer.best_cut_point['cut_b']
        )
        print_summary(optimized_hybrid, "Optimized Hybrid")
        
        # Train adapter
        print("Training optimized adapter...")
        train_only_adapter(optimized_hybrid)
        train_model(optimized_hybrid, epochs=15, dataloader=train_loader)
        
        # Evaluate
        optimized_acc = evaluate(optimized_hybrid, test_loader)
        print(f"Optimized Accuracy: {optimized_acc*100:.2f}%")
        
        # Comparison
        print("\n" + "="*80)
        print("RESULTS COMPARISON")
        print("="*80)
        print(f"Baseline Hybrid:  {baseline_acc*100:.2f}%")
        print(f"Optimized Hybrid: {optimized_acc*100:.2f}%")
        print(f"Improvement:      {(optimized_acc - baseline_acc)*100:+.2f}%")
        
        if optimized_acc > baseline_acc:
            print("\n✓ Layer matching analysis successfully improved hybrid performance!")
        else:
            print("\n⚠ Optimized hybrid didn't outperform baseline.")
            print("  This could mean:")
            print("  - More training epochs needed")
            print("  - Different hyperparameters required")
            print("  - Baseline cut point was already near-optimal")
        
        # Save comparison plot
        fig, ax = plt.subplots(figsize=(10, 6))
        configs = ['Baseline\n(Cut in Half)', f'Optimized\n(A:{analyzer.best_cut_point["cut_a"]}, B:{analyzer.best_cut_point["cut_b"]})']
        accuracies = [baseline_acc * 100, optimized_acc * 100]
        colors = ['#3498db', '#2ecc71' if optimized_acc > baseline_acc else '#e74c3c']
        
        bars = ax.bar(configs, accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title('Hybrid Model Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{acc:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # Add improvement annotation
        if optimized_acc != baseline_acc:
            mid_x = 0.5
            mid_y = (accuracies[0] + accuracies[1]) / 2
            improvement = (optimized_acc - baseline_acc) * 100
            color = 'green' if improvement > 0 else 'red'
            ax.annotate(f'{improvement:+.2f}%', 
                       xy=(mid_x, mid_y), fontsize=14, fontweight='bold',
                       color=color, ha='center')
        
        plt.tight_layout()
        output_path = Path(output_dir)
        plt.savefig(output_path / 'hybrid_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n✓ Comparison plot saved to {output_path / 'hybrid_comparison.png'}")


def main():
    """Main execution."""
    print("\n" + "="*80)
    print("HYBRID MODEL OPTIMIZATION USING LAYER MATCHING FRAMEWORK")
    print("="*80 + "\n")
    
    # Initialize analyzer
    analyzer = CNNLayerMatchAnalyzer(
        model_a_path='efficientnet.pth',
        model_b_path='biasnn.pth',
        device=DEVICE
    )
    
    # Analyze all cut points
    analyzer.analyze_all_cut_points()
    
    # Generate outputs
    analyzer.visualize_results(OUTPUT_DIR)
    analyzer.generate_report(OUTPUT_DIR)
    
    # Compare hybrids
    compare_hybrids(analyzer, OUTPUT_DIR)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"Results saved to: {OUTPUT_DIR}/")
    print("  • cut_point_heatmap.png - Match scores for all configurations")
    print("  • best_cut_point_analysis.png - Detailed analysis of optimal cut")
    print("  • top_cut_points_comparison.png - Top 5 configurations compared")
    print("  • hybrid_comparison.png - Performance comparison")
    print("  • hybrid_optimization_report.txt - Detailed text report")


if __name__ == "__main__":
    main()
