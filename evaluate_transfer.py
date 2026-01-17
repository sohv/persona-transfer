#!/usr/bin/env python3
"""
Evaluate cross-model persona transfer with systematic testing.

Usage:
    python evaluate_transfer.py --config experiments/cross_family_eval.yaml
    python evaluate_transfer.py --source qwen2.5-7b-instruct --target llama-3.1-8b-instruct \\
        --trait silly --coefficients -2.0 -1.0 0.0 1.0 2.0
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from models import _load_model, _generate_text_response, unload_model
from persona_vectors import load_persona_vectors
from prompts import get_evaluation_questions

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def compute_coherence(text):
    """Simple coherence scoring based on text quality indicators."""
    if not text or len(text.strip()) < 10:
        return 0.0
    
    score = 50.0  # Base score
    
    # Penalize gibberish indicators
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


async def evaluate_single_transfer(
    source_model_id,
    target_model_id,
    trait,
    vector_file,
    prompts,
    coefficients,
    max_tokens=150
):
    """Evaluate transfer for one source-target pair."""
    
    logger.info(f"\nEvaluating: {source_model_id} → {target_model_id}")
    logger.info(f"Trait: {trait}")
    logger.info(f"Coefficients: {coefficients}")
    logger.info(f"Test prompts: {len(prompts)}")
    
    # Load vectors
    vector_data = load_persona_vectors(Path(vector_file))
    if not vector_data:
        raise ValueError(f"Failed to load vectors from {vector_file}")
    
    # Load target model
    logger.info(f"Loading target model: {target_model_id}")
    model, tokenizer = _load_model(target_model_id)
    
    results = []
    total_tests = len(prompts) * len(coefficients)
    completed = 0
    
    start_time = time.time()
    
    # Test each prompt with each coefficient
    for prompt in prompts:
        prompt_results = {
            'prompt': prompt,
            'responses': []
        }
        
        for coefficient in coefficients:
            completed += 1
            logger.info(f"Progress: {completed}/{total_tests} - Testing coefficient {coefficient}")
            
            try:
                response = _generate_text_response(
                    model=model,
                    tokenizer=tokenizer,
                    model_id=target_model_id,
                    system_prompt="You are a helpful assistant.",
                    user_prompt=prompt,
                    max_new_tokens=max_tokens,
                    persona_vectors=vector_data['vectors'] if coefficient != 0.0 else None,
                    steering_coefficient=coefficient
                )
                
                coherence = compute_coherence(response)
                
                prompt_results['responses'].append({
                    'coefficient': coefficient,
                    'response': response,
                    'coherence': coherence,
                    'length': len(response.split())
                })
                
                logger.debug(f"  Response length: {len(response.split())} words, Coherence: {coherence:.1f}")
                
            except Exception as e:
                logger.error(f"  Failed: {e}")
                prompt_results['responses'].append({
                    'coefficient': coefficient,
                    'response': None,
                    'error': str(e),
                    'coherence': 0.0
                })
        
        results.append(prompt_results)
    
    elapsed = time.time() - start_time
    logger.info(f"Evaluation completed in {elapsed:.1f}s")
    
    # Unload model to free memory
    unload_model(target_model_id)
    
    return {
        'source_model': source_model_id,
        'target_model': target_model_id,
        'trait': trait,
        'vector_file': str(vector_file),
        'coefficients': coefficients,
        'num_prompts': len(prompts),
        'total_tests': total_tests,
        'elapsed_seconds': elapsed,
        'timestamp': datetime.now().isoformat(),
        'results': results
    }


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate cross-model persona transfer systematically',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Evaluate Qwen→Llama transfer:
    python evaluate_transfer.py \\
        --source qwen2.5-7b-instruct \\
        --target llama-3.1-8b-instruct \\
        --trait silly \\
        --coefficients -2.0 -1.0 0.0 1.0 2.0 \\
        --output experiments/qwen_to_llama_silly.json

  Quick 3-point test:
    python evaluate_transfer.py \\
        --source qwen2.5-7b-instruct \\
        --target mistral-7b-instruct-v0.3 \\
        --trait honest \\
        --coefficients -1.0 0.0 1.0 \\
        --num-prompts 5

  Full 9-point spectrum:
    python evaluate_transfer.py \\
        --source llama-3.1-8b-instruct \\
        --target qwen2.5-7b-instruct \\
        --trait silly \\
        --coefficients -2.0 -1.5 -1.0 -0.5 0.0 0.5 1.0 1.5 2.0
        """
    )
    
    parser.add_argument(
        '--source',
        required=True,
        help='Source model ID (where vectors were extracted from)'
    )
    
    parser.add_argument(
        '--target',
        required=True,
        help='Target model ID (to apply steering to)'
    )
    
    parser.add_argument(
        '--trait',
        required=True,
        help='Trait being evaluated (e.g., silly, honest)'
    )
    
    parser.add_argument(
        '--vectors',
        help='Path to vector file (default: auto-detect from source/trait)'
    )
    
    parser.add_argument(
        '--coefficients',
        nargs='+',
        type=float,
        default=[-2.0, -1.0, 0.0, 1.0, 2.0],
        help='Steering coefficients to test (default: -2.0 -1.0 0.0 1.0 2.0)'
    )
    
    parser.add_argument(
        '--num-prompts',
        type=int,
        help='Number of test prompts to use (default: all available)'
    )
    
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=150,
        help='Max tokens per generation (default: 150)'
    )
    
    parser.add_argument(
        '--output',
        help='Output JSON file for results (default: experiments/TIMESTAMP.json)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Determine vector file
    if args.vectors:
        vector_file = Path(args.vectors)
    else:
        vector_file = Path(f"src/data/vectors/{args.source}_{args.trait}.json")
    
    if not vector_file.exists():
        logger.error(f"Vector file not found: {vector_file}")
        logger.info(f"Extract vectors first with: python extract_vectors.py --model {args.source} --trait {args.trait}")
        sys.exit(1)
    
    # Get test prompts
    prompts = get_evaluation_questions(args.trait)
    if not prompts:
        logger.error(f"No evaluation questions found for trait: {args.trait}")
        sys.exit(1)
    
    if args.num_prompts:
        prompts = prompts[:args.num_prompts]
    
    # Run evaluation
    logger.info("\n" + "="*80)
    logger.info("CROSS-MODEL TRANSFER EVALUATION")
    logger.info("="*80)
    
    try:
        evaluation_results = asyncio.run(
            evaluate_single_transfer(
                source_model_id=args.source,
                target_model_id=args.target,
                trait=args.trait,
                vector_file=vector_file,
                prompts=prompts,
                coefficients=args.coefficients,
                max_tokens=args.max_tokens
            )
        )
        
        # Determine output file
        if args.output:
            output_file = Path(args.output)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = Path(f"experiments/{args.source}_to_{args.target}_{args.trait}_{timestamp}.json")
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save results
        with open(output_file, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        logger.info(f"\nResults saved to: {output_file}")
        
        # Print summary
        logger.info("\n" + "="*80)
        logger.info("SUMMARY")
        logger.info("="*80)
        
        avg_coherence_by_coef = {}
        for prompt_result in evaluation_results['results']:
            for response in prompt_result['responses']:
                coef = response['coefficient']
                coherence = response.get('coherence', 0.0)
                if coef not in avg_coherence_by_coef:
                    avg_coherence_by_coef[coef] = []
                avg_coherence_by_coef[coef].append(coherence)
        
        logger.info("Average coherence by coefficient:")
        for coef in sorted(avg_coherence_by_coef.keys()):
            scores = avg_coherence_by_coef[coef]
            avg = sum(scores) / len(scores) if scores else 0.0
            logger.info(f"  {coef:+.1f}: {avg:.1f}/100")
        
        logger.info(f"\nTotal runtime: {evaluation_results['elapsed_seconds']:.1f}s")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
