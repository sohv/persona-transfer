#!/usr/bin/env python3
"""
Apply persona vector steering to generate text responses.

Usage:
    python apply_steering.py --model llama-3.1-8b-instruct \\
        --vectors src/data/vectors/qwen2.5-7b-instruct_silly.json \\
        --prompt "Explain how photosynthesis works" \\
        --coefficient 1.5

    python apply_steering.py --model mistral-7b-instruct-v0.3 \\
        --vectors vectors/honest.json \\
        --prompt "Tell me about the moon landing" \\
        --coefficient -1.0 --baseline
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from models import _load_model, _generate_text_response
from persona_vectors import load_persona_vectors_from_file

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Apply persona vector steering to generate responses',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Test cross-model transfer (Qwen vectors on Llama):
    python apply_steering.py --model llama-3.1-8b-instruct \\
        --vectors src/data/vectors/qwen2.5-7b-instruct_silly.json \\
        --prompt "Explain quantum physics" --coefficient 1.5

  Compare baseline vs steered:
    python apply_steering.py --model mistral-7b-instruct-v0.3 \\
        --vectors vectors/honest.json \\
        --prompt "What do you think about AI?" \\
        --coefficient 1.0 --baseline

  Negative steering (opposite trait):
    python apply_steering.py --model qwen2.5-7b-instruct \\
        --vectors src/data/vectors/qwen2.5-7b-instruct_silly.json \\
        --prompt "Describe a cat" --coefficient -2.0

  Save output to file:
    python apply_steering.py --model llama-3.1-8b-instruct \\
        --vectors vectors/silly.json \\
        --prompt "Explain computers" --coefficient 1.5 \\
        --output results.json
        """
    )
    
    parser.add_argument(
        '--model',
        required=True,
        help='Target model to apply steering to'
    )
    
    parser.add_argument(
        '--vectors',
        required=True,
        help='Path to persona vector file (JSON)'
    )
    
    parser.add_argument(
        '--prompt',
        required=True,
        help='Input prompt/question'
    )
    
    parser.add_argument(
        '--coefficient',
        type=float,
        default=1.0,
        help='Steering strength (-2.0 to 2.0, default: 1.0)'
    )
    
    parser.add_argument(
        '--baseline',
        action='store_true',
        help='Also generate baseline response (coefficient=0.0) for comparison'
    )
    
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=150,
        help='Maximum tokens to generate (default: 150)'
    )
    
    parser.add_argument(
        '--output',
        help='Save results to JSON file'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate coefficient
    if not -2.0 <= args.coefficient <= 2.0:
        logger.warning(f"Coefficient {args.coefficient} outside typical range [-2.0, 2.0]")
    
    # Load vectors
    logger.info(f"Loading persona vectors from: {args.vectors}")
    try:
        vector_data = load_persona_vectors_from_file(Path(args.vectors))
        if not vector_data:
            logger.error("Failed to load vectors")
            sys.exit(1)
        logger.info(f"Loaded vectors for {len(vector_data['vectors'])} layers")
    except Exception as e:
        logger.error(f"Failed to load vectors: {e}")
        sys.exit(1)
    
    # Load model
    logger.info(f"Loading model: {args.model}")
    try:
        model, tokenizer = _load_model(args.model)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)
    
    results = {}
    
    # Generate baseline if requested
    if args.baseline:
        logger.info("\nGenerating baseline response (no steering)...")
        try:
            baseline_response = _generate_text_response(
                model=model,
                tokenizer=tokenizer,
                model_id=args.model,
                system_prompt="You are a helpful assistant.",
                user_prompt=args.prompt,
                max_new_tokens=args.max_tokens,
                persona_vectors=None,
                steering_coefficient=0.0
            )
            results['baseline'] = {
                'coefficient': 0.0,
                'response': baseline_response
            }
            print("\n" + "="*80)
            print("BASELINE (coefficient=0.0):")
            print("="*80)
            print(baseline_response)
            print("="*80)
        except Exception as e:
            logger.error(f"Baseline generation failed: {e}")
    
    # Generate steered response
    logger.info(f"\nGenerating steered response (coefficient={args.coefficient})...")
    try:
        steered_response = _generate_text_response(
            model=model,
            tokenizer=tokenizer,
            model_id=args.model,
            system_prompt="You are a helpful assistant.",
            user_prompt=args.prompt,
            max_new_tokens=args.max_tokens,
            persona_vectors=vector_data['vectors'],
            steering_coefficient=args.coefficient
        )
        results['steered'] = {
            'coefficient': args.coefficient,
            'response': steered_response
        }
        print("\n" + "="*80)
        print(f"STEERED (coefficient={args.coefficient}):")
        print("="*80)
        print(steered_response)
        print("="*80)
    except Exception as e:
        logger.error(f"Steered generation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Save to file if requested
    if args.output:
        output_data = {
            'model': args.model,
            'vectors': args.vectors,
            'prompt': args.prompt,
            'max_tokens': args.max_tokens,
            'results': results
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"\nResults saved to: {args.output}")
    
    logger.info("\nDone!")


if __name__ == "__main__":
    main()
