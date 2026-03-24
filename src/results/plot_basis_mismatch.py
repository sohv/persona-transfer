#!/usr/bin/env python3
"""
Plot basis mismatch test results.

Shows that basis mismatch exists but is not the primary limiting factor in persona transfer.
The transferred direction has only modest similarity to native directions.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_basis_mismatch_results(results_file):
    """Generate plots for basis mismatch analysis."""
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    results = data['results_by_transfer']
    
    # Extract data for plotting
    transfer_pairs = [f"{r['source_model'].split('-')[0]} → {r['target_model'].split('-')[0]}" 
                      for r in results]
    cosine_sim_transferred = [r['metrics']['cosine_similarity_transferred_native'] for r in results]
    cosine_sim_random = [r['metrics']['cosine_similarity_to_random'] for r in results]
    native_strength = [r['metrics']['native_vector_trait_strength'] for r in results]
    transferred_strength = [r['metrics']['transferred_vector_trait_strength'] for r in results]
    ratios = [r['metrics']['ratio'] for r in results]
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Basis Mismatch: Direction Transfer Quality', fontsize=16, fontweight='bold')
    
    x = np.arange(len(transfer_pairs))
    width = 0.35
    
    # Plot 1: Cosine Similarity
    ax1 = axes[0]
    bars1 = ax1.bar(x - width/2, cosine_sim_transferred, width, label='Transferred vs Native', color='#9b59b6')
    bars2 = ax1.bar(x + width/2, cosine_sim_random, width, label='Random Direction', color='#95a5a6')
    ax1.axhline(y=0.3, color='red', linestyle='--', linewidth=2, label='Success Threshold (0.3)')
    ax1.set_ylabel('Cosine Similarity', fontweight='bold')
    ax1.set_title('Direction Similarity: Transferred vs Random')
    ax1.set_xticks(x)
    ax1.set_xticklabels(transfer_pairs, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim([0, 0.5])
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Native vs Transferred Trait Strength
    ax2 = axes[1]
    bars3 = ax2.bar(x - width/2, native_strength, width, label='Native Vector', color='#27ae60')
    bars4 = ax2.bar(x + width/2, transferred_strength, width, label='Transferred Vector', color='#e67e22')
    ax2.set_ylabel('Trait Strength (0-10)', fontweight='bold')
    ax2.set_title('Trait Expression: Native vs Transferred Direction')
    ax2.set_xticks(x)
    ax2.set_xticklabels(transfer_pairs, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim([0, 10])
    
    # Add value labels on bars
    for bars in [bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 3: Transfer Effectiveness Ratio
    ax3 = axes[2]
    colors = ['#27ae60' if r > 0.55 else '#e74c3c' for r in ratios]
    bars5 = ax3.bar(transfer_pairs, ratios, color=colors)
    ax3.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='50% Effectiveness')
    ax3.axhline(y=0.7, color='green', linestyle='--', linewidth=2, label='70% Effectiveness')
    ax3.set_ylabel('Effectiveness Ratio', fontweight='bold')
    ax3.set_title('Transferred Direction Effectiveness (Transferred/Native)')
    ax3.set_xticklabels(transfer_pairs, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    ax3.set_ylim([0, 1.0])
    
    # Add value labels on bars
    for bar in bars5:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0%}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(results_file).parent / 'basis_mismatch_plot.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_path}")
    plt.close()

if __name__ == '__main__':
    results_file = Path(__file__).parent / 'basis_mismatch_results.json'
    plot_basis_mismatch_results(results_file)
