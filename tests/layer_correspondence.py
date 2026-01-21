#!/usr/bin/env python3
"""
Hypothesis C: "The vector hits the wrong layer/function"

Test if layer roles differ across model families, causing functional mismatch.

Theory:
- Different architectures use layers for different purposes
- Layer 16 in Qwen might correspond to layer 20 in LLaMA functionally
- Injecting at the "wrong" layer hits incorrect computational stage
- Needs layer mapping: source_layer_i → target_layer_j

Test approach:
1. Systematically test each target layer with source vectors
2. For each layer: measure trait effect strength + perplexity/coherence
3. Find optimal layer correspondence by maximizing:
   - Trait expression (steering effect)
   - Minimal coherence degradation
4. Build a layer correspondence map

Success criteria:
- Find target layers that maintain coherence while showing trait effect
- Layer correspondence should be consistent (not random)
- Best layers should be in similar relative positions (e.g., both middle layers)
"""

import sys
import argparse
import asyncio
import json
import logging
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models import _load_model, _generate_text_response, unload_model
from persona_vectors import load_persona_vectors_from_file
from prompts import get_evaluation_questions

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def compute_coherence(text):
    """Simple coherence scoring based on text quality indicators."""
    if not text or len(text.strip()) < 10:
        return 0.0

    score = 50.0

    # Penalize gibberish
    if any(x in text.lower() for x in ['<unk>', '[unk]', '###', '�']):
        score -= 30

    # Reward complete sentences
    sentence_endings = text.count('.') + text.count('!') + text.count('?')
    score += min(sentence_endings * 5, 20)

    # Reward reasonable length
    word_count = len(text.split())
    if 20 <= word_count <= 200:
        score += 10
    elif word_count < 10:
        score -= 20

    # Reward proper capitalization
    if text[0].isupper():
        score += 5

    # Penalize excessive repetition
    words = text.lower().split()
    if len(words) > 0:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.5:
            score -= 20

    return max(0.0, min(100.0, score))


def estimate_trait_effect(baseline_response, steered_response, trait):
    """
    Estimate how much the trait is expressed in the steered response.
    Returns a score from 0-100.
    """
    # Simple heuristic: measure lexical differences
    baseline_words = set(baseline_response.lower().split())
    steered_words = set(steered_response.lower().split())

    # Measure novelty (new words introduced by steering)
    new_words = steered_words - baseline_words
    novelty_score = min(len(new_words) / max(len(baseline_words), 1) * 100, 50)

    # Trait-specific markers (expandable)
    trait_markers = {
        'silly': ['haha', 'lol', 'funny', 'joke', 'laugh', '!', 'hehe', 'silly'],
        'honest': ['honestly', 'truth', 'frankly', 'actually', 'fact', 'really'],
        'confident': ['definitely', 'certainly', 'absolutely', 'sure', 'confident'],
        'creative': ['imagine', 'create', 'novel', 'unique', 'innovative']
    }

    marker_score = 0
    if trait in trait_markers:
        markers_present = sum(1 for marker in trait_markers[trait] if marker in steered_response.lower())
        marker_score = min(markers_present * 10, 50)

    total_score = novelty_score + marker_score
    return min(total_score, 100)


def create_single_layer_vectors(source_vectors, target_layer_idx):
    """
    Create a vector dict that only contains one layer's vector.
    This allows testing individual layer injection.
    """
    # Find the corresponding source layer
    source_layer_name = f"layer_{target_layer_idx}"

    if source_layer_name in source_vectors:
        return {source_layer_name: source_vectors[source_layer_name]}

    # If exact match not found, try to find closest layer
    source_layer_indices = sorted([int(l.split('_')[1]) for l in source_vectors.keys() if 'layer_' in l])

    if not source_layer_indices:
        return {}

    # Use proportional mapping: map target layer to corresponding source layer
    # E.g., if target has 32 layers and source has 28, scale accordingly
    closest_idx = min(source_layer_indices, key=lambda x: abs(x - target_layer_idx))
    closest_layer_name = f"layer_{closest_idx}"

    if closest_layer_name in source_vectors:
        return {f"layer_{target_layer_idx}": source_vectors[closest_layer_name]}

    return {}


async def test_hypothesis_c(source_model, target_model, trait, source_vector_file, test_prompts, coefficient):
    """
    Test Hypothesis C: layer correspondence matters for cross-model transfer.
    """
    logger.info("\n" + "="*80)
    logger.info("HYPOTHESIS C: Layer Correspondence Test")
    logger.info("="*80)
    logger.info(f"Source: {source_model}")
    logger.info(f"Target: {target_model}")
    logger.info(f"Trait: {trait}")
    logger.info(f"Steering coefficient: {coefficient}")

    # Load source vectors
    source_vector_data = load_persona_vectors_from_file(source_vector_file)
    if not source_vector_data:
        raise ValueError(f"Failed to load source vectors from {source_vector_file}")

    source_vectors = source_vector_data['vectors']
    logger.info(f"Loaded {len(source_vectors)} source vectors")

    # Load target model
    logger.info(f"Loading target model: {target_model}")
    model, tokenizer = _load_model(target_model)

    # Determine number of layers in target model
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        num_target_layers = len(model.model.layers)
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        num_target_layers = len(model.transformer.h)
    else:
        logger.warning("Could not determine number of layers, assuming 32")
        num_target_layers = 32

    logger.info(f"Target model has {num_target_layers} layers")

    # Test each layer individually
    results = {
        'source_model': source_model,
        'target_model': target_model,
        'trait': trait,
        'coefficient': coefficient,
        'num_target_layers': num_target_layers,
        'layer_results': []
    }

    # Generate baseline responses (no steering)
    logger.info("\nGenerating baseline responses (no steering)...")
    baseline_responses = []

    for prompt in test_prompts:
        try:
            response = _generate_text_response(
                model=model,
                tokenizer=tokenizer,
                model_id=target_model,
                system_prompt="You are a helpful assistant.",
                user_prompt=prompt,
                max_new_tokens=100,
                persona_vectors=None,
                steering_coefficient=0.0
            )
            baseline_responses.append(response)
        except Exception as e:
            logger.error(f"  Baseline generation failed: {e}")
            baseline_responses.append("")

    # Test steering at different layers
    # Focus on key layers: early (0-25%), early-mid (25-40%), mid (40-60%), late-mid (60-75%), late (75-100%)
    test_layer_positions = [0.1, 0.25, 0.33, 0.5, 0.66, 0.75, 0.9]
    test_layers = [int(pos * (num_target_layers - 1)) for pos in test_layer_positions]

    logger.info(f"\nTesting layers: {test_layers}")

    for layer_idx in test_layers:
        logger.info(f"\n--- Testing Layer {layer_idx} ({layer_idx/num_target_layers*100:.0f}% depth) ---")

        # Create single-layer vectors
        layer_vectors = create_single_layer_vectors(source_vectors, layer_idx)

        if not layer_vectors:
            logger.warning(f"  Could not create vectors for layer {layer_idx}")
            continue

        layer_result = {
            'layer_index': layer_idx,
            'layer_position': layer_idx / num_target_layers,
            'prompt_results': []
        }

        # Test with this layer
        for i, prompt in enumerate(test_prompts):
            try:
                response = _generate_text_response(
                    model=model,
                    tokenizer=tokenizer,
                    model_id=target_model,
                    system_prompt="You are a helpful assistant.",
                    user_prompt=prompt,
                    max_new_tokens=100,
                    persona_vectors=layer_vectors,
                    steering_coefficient=coefficient
                )

                coherence = compute_coherence(response)
                trait_effect = estimate_trait_effect(baseline_responses[i], response, trait)

                layer_result['prompt_results'].append({
                    'prompt': prompt,
                    'baseline': baseline_responses[i],
                    'steered': response,
                    'coherence': coherence,
                    'trait_effect': trait_effect
                })

                logger.info(f"  Prompt {i+1}: coherence={coherence:.1f}, trait_effect={trait_effect:.1f}")

            except Exception as e:
                logger.error(f"  Generation failed: {e}")
                layer_result['prompt_results'].append({
                    'prompt': prompt,
                    'error': str(e),
                    'coherence': 0.0,
                    'trait_effect': 0.0
                })

        # Compute layer statistics
        coherences = [r['coherence'] for r in layer_result['prompt_results'] if 'coherence' in r]
        trait_effects = [r['trait_effect'] for r in layer_result['prompt_results'] if 'trait_effect' in r]

        layer_result['avg_coherence'] = np.mean(coherences) if coherences else 0.0
        layer_result['avg_trait_effect'] = np.mean(trait_effects) if trait_effects else 0.0

        # Combined score: balance coherence and trait effect
        # Good layer should have high coherence + strong trait effect
        layer_result['combined_score'] = (
            0.6 * layer_result['avg_coherence'] +
            0.4 * layer_result['avg_trait_effect']
        )

        results['layer_results'].append(layer_result)

        logger.info(f"  Layer {layer_idx} summary: coherence={layer_result['avg_coherence']:.1f}, "
                   f"trait_effect={layer_result['avg_trait_effect']:.1f}, "
                   f"combined_score={layer_result['combined_score']:.1f}")

    # Unload model
    unload_model(target_model)

    # Analysis: Find best layer
    logger.info("\n" + "="*80)
    logger.info("RESULTS SUMMARY")
    logger.info("="*80)

    if not results['layer_results']:
        logger.error("No layer results to analyze")
        return results

    sorted_layers = sorted(
        results['layer_results'],
        key=lambda x: x['combined_score'],
        reverse=True
    )

    logger.info("\nLayer Performance Ranking:")
    for i, layer_data in enumerate(sorted_layers[:5], 1):
        logger.info(
            f"{i}. Layer {layer_data['layer_index']} ({layer_data['layer_position']*100:.0f}% depth): "
            f"combined={layer_data['combined_score']:.1f} "
            f"(coherence={layer_data['avg_coherence']:.1f}, "
            f"trait={layer_data['avg_trait_effect']:.1f})"
        )

    best_layer = sorted_layers[0]
    worst_layer = sorted_layers[-1]

    # Hypothesis verdict
    # If best layer significantly outperforms worst layer, layer correspondence matters
    performance_gap = best_layer['combined_score'] - worst_layer['combined_score']
    hypothesis_supported = performance_gap > 15.0  # 15 point threshold

    results['hypothesis_verdict'] = {
        'supported': hypothesis_supported,
        'best_layer': best_layer['layer_index'],
        'best_layer_position': best_layer['layer_position'],
        'performance_gap': performance_gap,
        'conclusion': (
            f"{'SUPPORTED' if hypothesis_supported else 'NOT SUPPORTED'}: "
            f"Best layer (L{best_layer['layer_index']}) outperforms worst by {performance_gap:.1f} points. "
            f"{'Layer correspondence is critical.' if hypothesis_supported else 'Layer choice has minimal impact.'}"
        )
    }

    logger.info("\n" + "="*80)
    logger.info(f"HYPOTHESIS C: {results['hypothesis_verdict']['conclusion']}")
    logger.info("="*80)

    # Recommended layer mapping
    if hypothesis_supported:
        logger.info(f"\nRECOMMENDED LAYER MAPPING:")
        logger.info(f"  {source_model} → {target_model}")
        logger.info(f"  Use target layer {best_layer['layer_index']} ({best_layer['layer_position']*100:.0f}% depth)")

    return results


def main():
    parser = argparse.ArgumentParser(description='Test Hypothesis C: Layer Correspondence')
    parser.add_argument('--source', required=True, help='Source model ID')
    parser.add_argument('--target', required=True, help='Target model ID')
    parser.add_argument('--trait', required=True, help='Trait name')
    parser.add_argument('--vectors', help='Path to source vector file (auto-detect if not provided)')
    parser.add_argument('--coefficient', type=float, default=2.0,
                       help='Steering coefficient to test (default: 2.0)')
    parser.add_argument('--num-prompts', type=int, default=5, help='Number of test prompts')
    parser.add_argument('--output', help='Output JSON file')

    args = parser.parse_args()

    # Determine vector file
    if args.vectors:
        vector_file = Path(args.vectors)
    else:
        vector_file = Path(f"src/data/vectors/{args.source}_{args.trait}.json")

    if not vector_file.exists():
        logger.error(f"Vector file not found: {vector_file}")
        sys.exit(1)

    # Get test prompts
    prompts = get_evaluation_questions(args.trait)
    if not prompts:
        logger.error(f"No evaluation questions for trait: {args.trait}")
        sys.exit(1)

    prompts = prompts[:args.num_prompts]

    # Run test
    results = asyncio.run(test_hypothesis_c(
        source_model=args.source,
        target_model=args.target,
        trait=args.trait,
        source_vector_file=vector_file,
        test_prompts=prompts,
        coefficient=args.coefficient
    ))

    # Save results
    if args.output:
        output_file = Path(args.output)
    else:
        output_file = Path(f"experiments/hypothesis_c_{args.source}_to_{args.target}_{args.trait}.json")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
