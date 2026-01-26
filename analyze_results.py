#!/usr/bin/env python3
"""
Aggregate evaluation results and generate paper tables/figures.

Usage:
    python analyze_results.py --results experiments/*.json
    python analyze_results.py --results experiments/ --output paper_results/
"""

import argparse
import json
import numpy as np
import sys
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any

def load_results(result_files: List[Path]) -> List[Dict[str, Any]]:
    """Load all result JSON files."""
    results = []
    for file_path in result_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                results.append(data)
        except Exception as e:
            print(f"Warning: Failed to load {file_path}: {e}", file=sys.stderr)
    return results


def extract_metrics_by_coefficient(result: Dict[str, Any]) -> Dict[float, Dict[str, List[float]]]:
    """Extract metrics organized by coefficient."""
    metrics_by_coef = defaultdict(lambda: {
        'coherence': [],
        'trait_strength': [],
        'collapsed': []
    })

    for prompt_result in result.get('results', []):
        for response in prompt_result.get('responses', []):
            coef = response.get('coefficient', 0.0)
            coherence = response.get('coherence', 0.0)
            trait_strength = response.get('trait_strength')
            collapsed = response.get('collapsed', False)

            metrics_by_coef[coef]['coherence'].append(coherence)
            metrics_by_coef[coef]['collapsed'].append(1 if collapsed else 0)
            if trait_strength is not None:
                metrics_by_coef[coef]['trait_strength'].append(trait_strength)

    return metrics_by_coef


def compute_statistics(values: List[float]) -> Dict[str, float]:
    """Compute mean and std for a list of values."""
    if not values:
        return {'mean': 0.0, 'std': 0.0, 'count': 0}

    arr = np.array(values)
    return {
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr)),
        'count': len(values)
    }


def generate_transfer_quality_table(results: List[Dict[str, Any]]) -> str:
    """
    Generate Table 1: Transfer Quality by Model Pair.

    Columns: Source→Target | Trait | Coherence | Trait Strength | Collapse Rate
    """
    table_lines = []
    table_lines.append("=" * 100)
    table_lines.append("TABLE 1: Transfer Quality by Model Pair")
    table_lines.append("=" * 100)
    table_lines.append(f"{'Source→Target':<25} | {'Trait':<10} | {'Coherence':<15} | {'Trait Str.':<15} | {'Collapse %':<10}")
    table_lines.append("-" * 100)

    # Group by transfer pair
    for result in results:
        source = result.get('source_model', 'unknown')
        target = result.get('target_model', 'unknown')
        trait = result.get('trait', 'unknown')
        transfer_type = result.get('transfer_type', 'unknown')

        # Get metrics for baseline (coefficient = 0.0) and best steering (coefficient = 1.0 or 2.0)
        metrics_by_coef = extract_metrics_by_coefficient(result)

        # Average across all coefficients for summary
        all_coherence = []
        all_trait_strength = []
        all_collapsed = []

        for coef, metrics in metrics_by_coef.items():
            all_coherence.extend(metrics['coherence'])
            all_trait_strength.extend(metrics['trait_strength'])
            all_collapsed.extend(metrics['collapsed'])

        coherence_stats = compute_statistics(all_coherence)
        trait_stats = compute_statistics(all_trait_strength)
        collapse_rate = (sum(all_collapsed) / len(all_collapsed) * 100) if all_collapsed else 0.0

        transfer_pair = f"{source} → {target}"
        if source == target:
            transfer_pair += " (intra)"

        coherence_str = f"{coherence_stats['mean']:.1f} ± {coherence_stats['std']:.1f}"
        trait_str = f"{trait_stats['mean']:.1f} ± {trait_stats['std']:.1f}" if trait_stats['count'] > 0 else "N/A"

        table_lines.append(f"{transfer_pair:<25} | {trait:<10} | {coherence_str:<15} | {trait_str:<15} | {collapse_rate:<10.1f}")

    table_lines.append("=" * 100)
    return "\n".join(table_lines)


def generate_coefficient_sweep_table(results: List[Dict[str, Any]]) -> str:
    """
    Generate Table 2: Metrics by Coefficient Sweep.

    For each model pair, show how metrics change with coefficient.
    """
    table_lines = []
    table_lines.append("=" * 100)
    table_lines.append("TABLE 2: Coefficient Sweep Results")
    table_lines.append("=" * 100)

    for result in results:
        source = result.get('source_model', 'unknown')
        target = result.get('target_model', 'unknown')
        trait = result.get('trait', 'unknown')

        table_lines.append(f"\n{source} → {target} ({trait})")
        table_lines.append(f"{'Coef':<8} | {'Coherence':<15} | {'Trait Str.':<15} | {'Collapse %':<10}")
        table_lines.append("-" * 60)

        metrics_by_coef = extract_metrics_by_coefficient(result)

        for coef in sorted(metrics_by_coef.keys()):
            metrics = metrics_by_coef[coef]

            coherence_stats = compute_statistics(metrics['coherence'])
            trait_stats = compute_statistics(metrics['trait_strength'])
            collapse_rate = (sum(metrics['collapsed']) / len(metrics['collapsed']) * 100) if metrics['collapsed'] else 0.0

            coherence_str = f"{coherence_stats['mean']:.1f} ± {coherence_stats['std']:.1f}"
            trait_str = f"{trait_stats['mean']:.1f} ± {trait_stats['std']:.1f}" if trait_stats['count'] > 0 else "N/A"

            table_lines.append(f"{coef:+.1f}    | {coherence_str:<15} | {trait_str:<15} | {collapse_rate:<10.1f}")

    table_lines.append("=" * 100)
    return "\n".join(table_lines)


def generate_comparison_table(results: List[Dict[str, Any]]) -> str:
    """
    Generate Table 3: Intra-Model vs Cross-Model Comparison.

    Compare same trait transferred intra-model vs cross-model.
    """
    table_lines = []
    table_lines.append("=" * 100)
    table_lines.append("TABLE 3: Intra-Model vs Cross-Model Transfer Comparison")
    table_lines.append("=" * 100)
    table_lines.append(f"{'Trait':<10} | {'Metric':<15} | {'Intra-Model':<20} | {'Cross-Model':<20} | {'Δ (Cross-Intra)':<15}")
    table_lines.append("-" * 100)

    # Group by trait
    by_trait = defaultdict(lambda: {'intra': [], 'cross': []})

    for result in results:
        trait = result.get('trait', 'unknown')
        transfer_type = result.get('transfer_type', 'unknown')

        metrics_by_coef = extract_metrics_by_coefficient(result)

        # Aggregate across all coefficients
        all_coherence = []
        all_trait_strength = []
        all_collapsed = []

        for coef, metrics in metrics_by_coef.items():
            all_coherence.extend(metrics['coherence'])
            all_trait_strength.extend(metrics['trait_strength'])
            all_collapsed.extend(metrics['collapsed'])

        coherence_stats = compute_statistics(all_coherence)
        trait_stats = compute_statistics(all_trait_strength)
        collapse_rate = (sum(all_collapsed) / len(all_collapsed) * 100) if all_collapsed else 0.0

        category = 'intra' if transfer_type == 'same-model' else 'cross'
        by_trait[trait][category].append({
            'coherence': coherence_stats,
            'trait_strength': trait_stats,
            'collapse_rate': collapse_rate
        })

    # Generate comparison rows
    for trait in sorted(by_trait.keys()):
        intra_results = by_trait[trait]['intra']
        cross_results = by_trait[trait]['cross']

        if not intra_results or not cross_results:
            continue

        # Average across multiple runs of same type
        intra_coherence = np.mean([r['coherence']['mean'] for r in intra_results])
        cross_coherence = np.mean([r['coherence']['mean'] for r in cross_results])
        delta_coherence = cross_coherence - intra_coherence

        intra_trait = np.mean([r['trait_strength']['mean'] for r in intra_results if r['trait_strength']['count'] > 0])
        cross_trait = np.mean([r['trait_strength']['mean'] for r in cross_results if r['trait_strength']['count'] > 0])
        delta_trait = cross_trait - intra_trait

        intra_collapse = np.mean([r['collapse_rate'] for r in intra_results])
        cross_collapse = np.mean([r['collapse_rate'] for r in cross_results])
        delta_collapse = cross_collapse - intra_collapse

        table_lines.append(f"{trait:<10} | {'Coherence':<15} | {intra_coherence:<20.1f} | {cross_coherence:<20.1f} | {delta_coherence:+.1f}")
        table_lines.append(f"{'':10} | {'Trait Strength':<15} | {intra_trait:<20.1f} | {cross_trait:<20.1f} | {delta_trait:+.1f}")
        table_lines.append(f"{'':10} | {'Collapse %':<15} | {intra_collapse:<20.1f} | {cross_collapse:<20.1f} | {delta_collapse:+.1f}")
        table_lines.append("-" * 100)

    table_lines.append("=" * 100)
    return "\n".join(table_lines)


def generate_csv_export(results: List[Dict[str, Any]], output_dir: Path):
    """Export detailed CSV files for further analysis."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # CSV 1: Per-response data
    csv_path = output_dir / "detailed_responses.csv"
    with open(csv_path, 'w') as f:
        f.write("source,target,trait,coefficient,prompt_idx,coherence,trait_strength,collapsed,response_length\n")

        for result in results:
            source = result.get('source_model', '')
            target = result.get('target_model', '')
            trait = result.get('trait', '')

            for prompt_idx, prompt_result in enumerate(result.get('results', [])):
                for response in prompt_result.get('responses', []):
                    coef = response.get('coefficient', 0.0)
                    coherence = response.get('coherence', 0.0)
                    trait_strength = response.get('trait_strength', '')
                    collapsed = 1 if response.get('collapsed', False) else 0
                    length = response.get('length', 0)

                    f.write(f"{source},{target},{trait},{coef},{prompt_idx},{coherence},{trait_strength},{collapsed},{length}\n")

    print(f"Exported detailed CSV to: {csv_path}")

    # CSV 2: Aggregated by coefficient
    csv_path = output_dir / "aggregated_by_coefficient.csv"
    with open(csv_path, 'w') as f:
        f.write("source,target,trait,coefficient,coherence_mean,coherence_std,trait_strength_mean,trait_strength_std,collapse_rate\n")

        for result in results:
            source = result.get('source_model', '')
            target = result.get('target_model', '')
            trait = result.get('trait', '')

            metrics_by_coef = extract_metrics_by_coefficient(result)

            for coef in sorted(metrics_by_coef.keys()):
                metrics = metrics_by_coef[coef]

                coherence_stats = compute_statistics(metrics['coherence'])
                trait_stats = compute_statistics(metrics['trait_strength'])
                collapse_rate = (sum(metrics['collapsed']) / len(metrics['collapsed']) * 100) if metrics['collapsed'] else 0.0

                f.write(f"{source},{target},{trait},{coef},{coherence_stats['mean']:.2f},{coherence_stats['std']:.2f},")
                f.write(f"{trait_stats['mean']:.2f},{trait_stats['std']:.2f},{collapse_rate:.2f}\n")

    print(f"Exported aggregated CSV to: {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze evaluation results and generate paper tables/figures',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Analyze all results in experiments directory:
    python analyze_results.py --results experiments/*.json

  Generate tables and export CSVs:
    python analyze_results.py --results experiments/ --output paper_results/
        """
    )

    parser.add_argument(
        '--results',
        nargs='+',
        required=True,
        help='Result JSON files or directory containing results'
    )

    parser.add_argument(
        '--output',
        type=Path,
        default=Path('analysis_output'),
        help='Output directory for generated tables/CSVs (default: analysis_output/)'
    )

    args = parser.parse_args()

    # Collect result files
    result_files = []
    for path_str in args.results:
        path = Path(path_str)
        if path.is_dir():
            result_files.extend(path.glob('*.json'))
        elif path.is_file() and path.suffix == '.json':
            result_files.append(path)
        else:
            # Handle glob patterns
            result_files.extend(Path('.').glob(path_str))

    if not result_files:
        print("Error: No result files found", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {len(result_files)} result files...")
    results = load_results(result_files)

    if not results:
        print("Error: No valid results loaded", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(results)} experiments\n")

    # Generate tables
    print(generate_transfer_quality_table(results))
    print("\n")
    print(generate_coefficient_sweep_table(results))
    print("\n")
    print(generate_comparison_table(results))

    # Export CSVs
    print(f"\nExporting detailed data to {args.output}/...")
    generate_csv_export(results, args.output)

    # Save tables to file
    tables_file = args.output / "tables.txt"
    with open(tables_file, 'w') as f:
        f.write(generate_transfer_quality_table(results))
        f.write("\n\n")
        f.write(generate_coefficient_sweep_table(results))
        f.write("\n\n")
        f.write(generate_comparison_table(results))

    print(f"Saved tables to: {tables_file}")
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
