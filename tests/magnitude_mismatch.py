#!/usr/bin/env python3
"""
Hypothesis A: "The direction is meaningful, but scaled wrong"

Test if cross-model activation norms differ, causing magnitude mismatch.

Theory:
- The persona vector direction is correct
- But the magnitude is too large/small for the target model
- Target model activations operate at different scales

Test approach:
1. Normalize steering vectors by target model's activation norm
2. Use: v_target = v / ||v||
3. Apply α * std(residual_stream_layer) scaling
4. Measure if coherence improves with normalization

Success criteria:
- Normalized vectors should produce better coherence than raw vectors
- Optimal scaling factor should be consistent across layers
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


def compute_activation_stats(model, tokenizer, model_id, test_prompts):
    """
    Compute activation statistics for the target model.
    Returns mean and std of activations across layers.
    """
    logger.info("Computing target model activation statistics...")

    activation_stats = {}

    # Generate responses for neutral prompts to capture baseline activations
    from models import _attach_activation_hooks, _detach_hooks

    for prompt in test_prompts[:3]:  # Use first 3 prompts for stats
        activations, hooks = _attach_activation_hooks(model, model_id)

        try:
            _ = _generate_text_response(
                model=model,
                tokenizer=tokenizer,
                model_id=model_id,
                system_prompt="You are a helpful assistant.",
                user_prompt=prompt,
                max_new_tokens=50,
                persona_vectors=None,
                steering_coefficient=0.0
            )

            # Collect activation statistics
            for layer_name, activation in activations.items():
                if layer_name not in activation_stats:
                    activation_stats[layer_name] = []
                activation_stats[layer_name].append(activation)

        finally:
            _detach_hooks(hooks)

    # Compute mean and std for each layer
    stats = {}
    for layer_name, activations_list in activation_stats.items():
        activations_array = np.concatenate(activations_list, axis=0)
        stats[layer_name] = {
            'mean': np.mean(activations_array),
            'std': np.std(activations_array),
            'norm': np.mean([np.linalg.norm(a) for a in activations_list])
        }
        logger.info(f"  {layer_name}: mean={stats[layer_name]['mean']:.4f}, "
                   f"std={stats[layer_name]['std']:.4f}, norm={stats[layer_name]['norm']:.4f}")

    return stats


def normalize_vectors(vectors, target_stats, method='unit_norm'):
    """
    Normalize persona vectors based on target model statistics.

    Methods:
    - 'unit_norm': v_normalized = v / ||v||
    - 'target_norm': v_normalized = v * (target_norm / source_norm)
    - 'target_std': v_normalized = (v / ||v||) * target_std
    """
    normalized = {}

    for layer_name, vector in vectors.items():
        if not isinstance(vector, np.ndarray):
            continue

        source_norm = np.linalg.norm(vector)

        if method == 'unit_norm':
            # Simple unit normalization
            normalized[layer_name] = vector / source_norm if source_norm > 0 else vector

        elif method == 'target_norm' and layer_name in target_stats:
            # Scale to match target activation norm
            target_norm = target_stats[layer_name]['norm']
            normalized[layer_name] = vector * (target_norm / source_norm) if source_norm > 0 else vector

        elif method == 'target_std' and layer_name in target_stats:
            # Unit normalize then scale by target std
            unit_vector = vector / source_norm if source_norm > 0 else vector
            target_std = target_stats[layer_name]['std']
            normalized[layer_name] = unit_vector * target_std

        else:
            # Fallback to raw vector
            normalized[layer_name] = vector

    return normalized


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


async def test_hypothesis_a(source_model, target_model, trait, vector_file, coefficients, test_prompts):
    """
    Test Hypothesis A: magnitude mismatch causes coherence issues.
    """
    logger.info("\n" + "="*80)
    logger.info("HYPOTHESIS A: Magnitude Mismatch Test")
    logger.info("="*80)
    logger.info(f"Source: {source_model}")
    logger.info(f"Target: {target_model}")
    logger.info(f"Trait: {trait}")

    # Load source vectors
    vector_data = load_persona_vectors_from_file(vector_file)
    if not vector_data:
        raise ValueError(f"Failed to load vectors from {vector_file}")

    source_vectors = vector_data['vectors']
    logger.info(f"Loaded {len(source_vectors)} source vectors")

    # Load target model
    logger.info(f"Loading target model: {target_model}")
    model, tokenizer = _load_model(target_model)

    # Compute target model activation statistics
    target_stats = compute_activation_stats(model, tokenizer, target_model, test_prompts)

    # Test different normalization methods
    normalization_methods = ['raw', 'unit_norm', 'target_norm', 'target_std']

    results = {
        'source_model': source_model,
        'target_model': target_model,
        'trait': trait,
        'normalization_results': {}
    }

    for norm_method in normalization_methods:
        logger.info(f"\n--- Testing normalization method: {norm_method} ---")

        # Prepare vectors
        if norm_method == 'raw':
            test_vectors = source_vectors
        else:
            test_vectors = normalize_vectors(source_vectors, target_stats, method=norm_method)

        method_results = {
            'method': norm_method,
            'coherence_scores': []
        }

        # Test with different coefficients
        for coef in coefficients:
            logger.info(f"  Testing coefficient: {coef}")

            prompt_scores = []
            for prompt in test_prompts:
                try:
                    response = _generate_text_response(
                        model=model,
                        tokenizer=tokenizer,
                        model_id=target_model,
                        system_prompt="You are a helpful assistant.",
                        user_prompt=prompt,
                        max_new_tokens=100,
                        persona_vectors=test_vectors if coef != 0.0 else None,
                        steering_coefficient=coef
                    )

                    coherence = compute_coherence(response)
                    prompt_scores.append(coherence)
                    logger.debug(f"    Prompt coherence: {coherence:.1f}")

                except Exception as e:
                    logger.error(f"    Generation failed: {e}")
                    prompt_scores.append(0.0)

            avg_coherence = np.mean(prompt_scores) if prompt_scores else 0.0
            method_results['coherence_scores'].append({
                'coefficient': coef,
                'avg_coherence': avg_coherence,
                'individual_scores': prompt_scores
            })
            logger.info(f"    Average coherence: {avg_coherence:.1f}")

        results['normalization_results'][norm_method] = method_results

    # Unload model
    unload_model(target_model)

    # Analysis
    logger.info("\n" + "="*80)
    logger.info("RESULTS SUMMARY")
    logger.info("="*80)

    for norm_method in normalization_methods:
        method_data = results['normalization_results'][norm_method]
        coherence_scores = [score['avg_coherence'] for score in method_data['coherence_scores']]
        avg_coherence = np.mean(coherence_scores)
        logger.info(f"{norm_method:15s}: avg coherence = {avg_coherence:.1f}")

    # Determine best method
    best_method = max(
        normalization_methods,
        key=lambda m: np.mean([s['avg_coherence'] for s in results['normalization_results'][m]['coherence_scores']])
    )

    logger.info(f"\nBest normalization method: {best_method}")

    # Hypothesis verdict
    raw_coherence = np.mean([s['avg_coherence'] for s in results['normalization_results']['raw']['coherence_scores']])
    best_coherence = np.mean([s['avg_coherence'] for s in results['normalization_results'][best_method]['coherence_scores']])

    improvement = best_coherence - raw_coherence
    results['hypothesis_verdict'] = {
        'supported': improvement > 5.0,  # 5 point improvement threshold
        'improvement': improvement,
        'conclusion': f"{'SUPPORTED' if improvement > 5.0 else 'NOT SUPPORTED'}: "
                     f"Normalization improved coherence by {improvement:.1f} points"
    }

    logger.info("\n" + "="*80)
    logger.info(f"HYPOTHESIS A: {results['hypothesis_verdict']['conclusion']}")
    logger.info("="*80)

    return results


def main():
    parser = argparse.ArgumentParser(description='Test Hypothesis A: Magnitude Mismatch')
    parser.add_argument('--source', required=True, help='Source model ID')
    parser.add_argument('--target', required=True, help='Target model ID')
    parser.add_argument('--trait', required=True, help='Trait name')
    parser.add_argument('--vectors', help='Path to vector file (auto-detect if not provided)')
    parser.add_argument('--coefficients', nargs='+', type=float, default=[-1.0, 0.0, 1.0, 2.0],
                       help='Steering coefficients to test')
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
    results = asyncio.run(test_hypothesis_a(
        source_model=args.source,
        target_model=args.target,
        trait=args.trait,
        vector_file=vector_file,
        coefficients=args.coefficients,
        test_prompts=prompts
    ))

    # Save results
    if args.output:
        output_file = Path(args.output)
    else:
        output_file = Path(f"experiments/hypothesis_a_{args.source}_to_{args.target}_{args.trait}.json")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
