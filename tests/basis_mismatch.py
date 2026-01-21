#!/usr/bin/env python3
"""
Hypothesis B: "The direction is rotated (basis mismatch)"

Test if the persona vector direction is meaningful in the target model's latent space.
Even if hidden sizes match, the basis might be completely different.

Theory:
- Each model learns different latent representations
- A vector meaningful in one model's space is random noise in another's
- The transferred vector points in the "wrong direction"

Test approach:
1. Extract "native" persona vectors from the target model (same trait prompts)
2. Compare transferred vector to native vector using cosine similarity
3. If cosine similarity ≈ 0, the direction is meaningless (basis mismatch)
4. If cosine similarity > 0.5, the direction is preserved (shared structure)

Success criteria:
- High cosine similarity (>0.5) indicates shared semantic structure
- Low cosine similarity (<0.2) indicates basis mismatch
- Can also test: applying native vectors should give better coherence than transferred
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
from persona_vectors import (
    load_persona_vectors_from_file,
    generate_persona_vectors
)
from prompts import get_prompt_pairs, get_evaluation_questions

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def compute_cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors."""
    # Handle dimension mismatch by truncating/padding
    if vec1.shape[0] != vec2.shape[0]:
        min_dim = min(vec1.shape[0], vec2.shape[0])
        vec1 = vec1[:min_dim]
        vec2 = vec2[:min_dim]

    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return np.dot(vec1, vec2) / (norm1 * norm2)


def compute_vector_similarity_matrix(source_vectors, target_vectors):
    """
    Compute cosine similarity matrix between source and target vectors.
    Returns dict of layer similarities.
    """
    similarities = {}

    # Find common layers
    source_layers = set(source_vectors.keys())
    target_layers = set(target_vectors.keys())

    # Try to match layers by index if exact names don't match
    source_layer_indices = sorted([int(l.split('_')[1]) for l in source_layers if 'layer_' in l])
    target_layer_indices = sorted([int(l.split('_')[1]) for l in target_layers if 'layer_' in l])

    logger.info(f"Source layers: {len(source_layer_indices)}, Target layers: {len(target_layer_indices)}")

    # Compare corresponding layers
    for src_idx, tgt_idx in zip(source_layer_indices, target_layer_indices):
        src_layer = f"layer_{src_idx}"
        tgt_layer = f"layer_{tgt_idx}"

        if src_layer in source_vectors and tgt_layer in target_vectors:
            src_vec = source_vectors[src_layer]
            tgt_vec = target_vectors[tgt_layer]

            if isinstance(src_vec, np.ndarray) and isinstance(tgt_vec, np.ndarray):
                similarity = compute_cosine_similarity(src_vec, tgt_vec)
                similarities[f"{src_layer}_to_{tgt_layer}"] = similarity
                logger.info(f"  {src_layer} → {tgt_layer}: similarity = {similarity:.4f}")

    return similarities


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


async def test_hypothesis_b(source_model, target_model, trait, source_vector_file, test_prompts, coefficients):
    """
    Test Hypothesis B: basis mismatch causes direction to be meaningless.
    """
    logger.info("\n" + "="*80)
    logger.info("HYPOTHESIS B: Basis Mismatch Test")
    logger.info("="*80)
    logger.info(f"Source: {source_model}")
    logger.info(f"Target: {target_model}")
    logger.info(f"Trait: {trait}")

    # Load source vectors (transferred)
    source_vector_data = load_persona_vectors_from_file(source_vector_file)
    if not source_vector_data:
        raise ValueError(f"Failed to load source vectors from {source_vector_file}")

    source_vectors = source_vector_data['vectors']
    logger.info(f"Loaded {len(source_vectors)} source vectors")

    # Generate native vectors for the target model
    logger.info("\nGenerating native persona vectors for target model...")
    logger.info("(This will take a few minutes)")

    prompt_pairs = get_prompt_pairs(trait)
    questions = get_evaluation_questions(trait)

    target_vector_data = await generate_persona_vectors(
        model_id=target_model,
        trait_id=trait,
        prompt_pairs=prompt_pairs,
        questions=questions[:5]  # Use subset for speed
    )

    target_vectors = target_vector_data['vectors']
    logger.info(f"Generated {len(target_vectors)} native target vectors")

    # Compute vector similarity
    logger.info("\n--- Computing Vector Similarity ---")
    similarities = compute_vector_similarity_matrix(source_vectors, target_vectors)

    if not similarities:
        logger.error("No layer similarities computed (dimension mismatch or no common layers)")
        return None

    avg_similarity = np.mean(list(similarities.values()))
    logger.info(f"\nAverage cosine similarity: {avg_similarity:.4f}")

    # Test coherence with transferred vs native vectors
    logger.info("\n--- Testing Coherence: Transferred vs Native ---")

    results = {
        'source_model': source_model,
        'target_model': target_model,
        'trait': trait,
        'vector_similarities': similarities,
        'avg_similarity': avg_similarity,
        'steering_tests': {
            'transferred': [],
            'native': []
        }
    }

    # Load target model once for testing
    logger.info(f"Loading target model: {target_model}")
    model, tokenizer = _load_model(target_model)

    # Test with transferred vectors
    logger.info("\nTesting transferred vectors...")
    for coef in coefficients:
        if coef == 0.0:
            continue

        logger.info(f"  Coefficient: {coef}")
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
                    persona_vectors=source_vectors,
                    steering_coefficient=coef
                )

                coherence = compute_coherence(response)
                prompt_scores.append(coherence)

            except Exception as e:
                logger.error(f"    Generation failed: {e}")
                prompt_scores.append(0.0)

        avg_coherence = np.mean(prompt_scores) if prompt_scores else 0.0
        results['steering_tests']['transferred'].append({
            'coefficient': coef,
            'avg_coherence': avg_coherence
        })
        logger.info(f"    Avg coherence: {avg_coherence:.1f}")

    # Test with native vectors
    logger.info("\nTesting native vectors...")
    for coef in coefficients:
        if coef == 0.0:
            continue

        logger.info(f"  Coefficient: {coef}")
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
                    persona_vectors=target_vectors,
                    steering_coefficient=coef
                )

                coherence = compute_coherence(response)
                prompt_scores.append(coherence)

            except Exception as e:
                logger.error(f"    Generation failed: {e}")
                prompt_scores.append(0.0)

        avg_coherence = np.mean(prompt_scores) if prompt_scores else 0.0
        results['steering_tests']['native'].append({
            'coefficient': coef,
            'avg_coherence': avg_coherence
        })
        logger.info(f"    Avg coherence: {avg_coherence:.1f}")

    # Unload model
    unload_model(target_model)

    # Analysis
    logger.info("\n" + "="*80)
    logger.info("RESULTS SUMMARY")
    logger.info("="*80)

    transferred_coherence = np.mean([t['avg_coherence'] for t in results['steering_tests']['transferred']])
    native_coherence = np.mean([t['avg_coherence'] for t in results['steering_tests']['native']])

    logger.info(f"Average cosine similarity: {avg_similarity:.4f}")
    logger.info(f"Transferred vectors coherence: {transferred_coherence:.1f}")
    logger.info(f"Native vectors coherence: {native_coherence:.1f}")
    logger.info(f"Coherence gap: {native_coherence - transferred_coherence:.1f}")

    # Hypothesis verdict
    # Low similarity + large coherence gap = basis mismatch
    basis_mismatch = avg_similarity < 0.3 and (native_coherence - transferred_coherence) > 10.0

    results['hypothesis_verdict'] = {
        'supported': basis_mismatch,
        'avg_similarity': avg_similarity,
        'coherence_gap': native_coherence - transferred_coherence,
        'conclusion': (
            f"{'SUPPORTED' if basis_mismatch else 'NOT SUPPORTED'}: "
            f"Cosine similarity = {avg_similarity:.4f}, "
            f"Native vectors {'significantly' if basis_mismatch else 'marginally'} "
            f"outperform transferred ({native_coherence:.1f} vs {transferred_coherence:.1f})"
        )
    }

    logger.info("\n" + "="*80)
    logger.info(f"HYPOTHESIS B: {results['hypothesis_verdict']['conclusion']}")
    logger.info("="*80)

    # Interpretation guide
    logger.info("\nInterpretation:")
    if avg_similarity > 0.5:
        logger.info("  → High similarity: Models share semantic structure")
    elif avg_similarity > 0.2:
        logger.info("  → Moderate similarity: Partial structure sharing")
    else:
        logger.info("  → Low similarity: Basis mismatch likely")

    return results


def main():
    parser = argparse.ArgumentParser(description='Test Hypothesis B: Basis Mismatch')
    parser.add_argument('--source', required=True, help='Source model ID')
    parser.add_argument('--target', required=True, help='Target model ID')
    parser.add_argument('--trait', required=True, help='Trait name')
    parser.add_argument('--vectors', help='Path to source vector file (auto-detect if not provided)')
    parser.add_argument('--coefficients', nargs='+', type=float, default=[1.0, 2.0],
                       help='Steering coefficients to test (default: 1.0 2.0)')
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
    results = asyncio.run(test_hypothesis_b(
        source_model=args.source,
        target_model=args.target,
        trait=args.trait,
        source_vector_file=vector_file,
        test_prompts=prompts,
        coefficients=args.coefficients
    ))

    if not results:
        logger.error("Test failed to produce results")
        sys.exit(1)

    # Save results
    if args.output:
        output_file = Path(args.output)
    else:
        output_file = Path(f"experiments/hypothesis_b_{args.source}_to_{args.target}_{args.trait}.json")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
