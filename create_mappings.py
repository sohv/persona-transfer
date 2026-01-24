#!/usr/bin/env python3
"""
Generate dimension mappings for cross-model persona vector transfer.

This script pre-computes Procrustes alignment mappings between models
with different hidden dimensions (e.g., Qwen 3584 ↔ LLaMA/Mistral 4096).

Usage:
    python create_mappings.py --all
    python create_mappings.py --source qwen2.5-7b-instruct --target llama-3.1-8b-instruct
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from itertools import combinations

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dimension_mapper import get_dimension_mapper

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_mapping_for_pair(source_model: str, target_model: str, num_samples: int = 50):
    """Create mapping for a single model pair."""

    # Load config to get dimensions
    config_path = Path("src/config/config.json")
    with open(config_path) as f:
        config = json.load(f)

    source_dim = config['models'][source_model]['hidden_size']
    target_dim = config['models'][target_model]['hidden_size']

    logger.info(f"\n{'='*80}")
    logger.info(f"Creating mapping: {source_model}({source_dim}) → {target_model}({target_dim})")
    logger.info(f"{'='*80}")

    # Skip if dimensions match
    if source_dim == target_dim:
        logger.info("Dimensions match, no mapping needed")
        return

    # Check if mapping already exists
    mapper = get_dimension_mapper()
    existing = mapper.load_mapping(source_model, target_model, source_dim, target_dim)
    if existing:
        logger.info(f"Mapping already exists (quality: cosine_sim={existing['quality_metrics']['cosine_similarity']:.4f})")
        return

    # Create new mapping
    try:
        mapping = mapper.create_mapping(
            source_model,
            target_model,
            source_dim,
            target_dim,
            num_samples=num_samples
        )
        logger.info(f"✓ Successfully created mapping")
        logger.info(f"  Quality: cosine_sim={mapping['quality_metrics']['cosine_similarity']:.4f}, "
                   f"mse={mapping['quality_metrics']['mse']:.6f}")
    except Exception as e:
        logger.error(f"✗ Failed to create mapping: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description='Generate dimension mappings for cross-model transfer'
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='Generate all necessary mappings between models with different dimensions'
    )

    parser.add_argument(
        '--source',
        help='Source model ID'
    )

    parser.add_argument(
        '--target',
        help='Target model ID'
    )

    parser.add_argument(
        '--num-samples',
        type=int,
        default=50,
        help='Number of aligned samples to collect (default: 50)'
    )

    args = parser.parse_args()

    # Load config
    config_path = Path("src/config/config.json")
    with open(config_path) as f:
        config = json.load(f)

    if args.all:
        # Generate all cross-model mappings where dimensions differ
        models = list(config['models'].keys())

        logger.info("Generating all necessary mappings...")
        logger.info(f"Models: {models}")

        # Find all pairs with different dimensions
        pairs_to_map = []
        for m1, m2 in combinations(models, 2):
            dim1 = config['models'][m1]['hidden_size']
            dim2 = config['models'][m2]['hidden_size']

            if dim1 != dim2:
                # Need bidirectional mappings
                pairs_to_map.append((m1, m2))
                pairs_to_map.append((m2, m1))

        logger.info(f"Need to create {len(pairs_to_map)} mappings")

        for source, target in pairs_to_map:
            try:
                create_mapping_for_pair(source, target, args.num_samples)
            except Exception as e:
                logger.error(f"Failed to create mapping {source}→{target}: {e}")
                continue

        logger.info("\n" + "="*80)
        logger.info("Mapping generation complete!")
        logger.info("="*80)

    elif args.source and args.target:
        # Generate single mapping
        create_mapping_for_pair(args.source, args.target, args.num_samples)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
