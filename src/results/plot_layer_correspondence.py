#!/usr/bin/env python3
"""
Plot layer correspondence test results.

Shows that layer correspondence is highly impactful in maintaining coherence
and maximizing trait expression during persona transfer.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_layer_correspondence_results(results_file):
    """Generate plots for layer correspondence analysis."""
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    results = data['results_by_transfer']
    
    # Extract data for plotting
    transfer_pairs = [f"{r['source_model'].split('-')[0]} → {r['target_model'].split('-')[0]}" 
                      for r in results]
    trait_strength_with_mapping = [r['metrics']['trait_strength_with_mapping'] for r in results]
    trait_strength_random = [r['metrics']['trait_strength_random_layer'] for r in results]
    coherence_with_mapping = [r['metrics']['coherence_with_mapping'] for r in results]
    coherence_random = [r['metrics']['coherence_random_layer'] for r in results]
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Layer Correspondence Impact on Persona Transfer', fontsize=16, fontweight='bold')
    
    x = np.arange(len(transfer_pairs))
    width = 0.35
    
    # Plot 1: Trait Strength Comparison
    ax1 = axes[0]
    bars1 = ax1.bar(x - width/2, trait_strength_with_mapping, width, label='With Layer Mapping', color='#2ecc71')
    bars2 = ax1.bar(x + width/2, trait_strength_random, width, label='Random Layer', color='#e74c3c')
    ax1.set_ylabel('Trait Strength (0-10)', fontweight='bold')
    ax1.set_title('Trait Expression: Mapped vs Random Layers')
    ax1.set_xticks(x)
    ax1.set_xticklabels(transfer_pairs, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim([0, 10])
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Coherence Comparison
    ax2 = axes[1]
    bars3 = ax2.bar(x - width/2, coherence_with_mapping, width, label='With Layer Mapping', color='#2ecc71')
    bars4 = ax2.bar(x + width/2, coherence_random, width, label='Random Layer', color='#e74c3c')
    ax2.set_ylabel('Coherence Score (0-1)', fontweight='bold')
    ax2.set_title('Coherence: Mapped vs Random Layers')
    ax2.set_xticks(x)
    ax2.set_xticklabels(transfer_pairs, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim([0, 1.0])
    
    # Add value labels on bars
    for bars in [bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 3: Improvement Percentage
    ax3 = axes[2]
    improvements = [r['metrics']['improvement'] for r in results]
    colors = ['#3498db' if imp > 250 else '#f39c12' for imp in improvements]
    bars5 = ax3.bar(transfer_pairs, improvements, color=colors)
    ax3.set_ylabel('Improvement (%)', fontweight='bold')
    ax3.set_title('Overall Improvement with Layer Mapping')
    ax3.set_xticklabels(transfer_pairs, rotation=45, ha='right')
    ax3.axhline(y=200, color='red', linestyle='--', linewidth=2, label='200% threshold')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars5:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(results_file).parent / 'layer_correspondence_plot.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_path}")
    plt.close()

if __name__ == '__main__':
    results_file = Path(__file__).parent / 'layer_correspondence_results.json'
    plot_layer_correspondence_results(results_file)
