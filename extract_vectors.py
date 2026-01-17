#!/usr/bin/env python3
"""
Extract persona vectors from a source model.

Usage:
    python extract_vectors.py --model qwen2.5-7b-instruct --trait silly
    python extract_vectors.py --model llama-3.1-8b-instruct --trait honest --output my_vectors/
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from models import _load_model
from persona_vectors import generate_persona_vectors, NumpyEncoder
from prompts import get_prompt_pairs, get_evaluation_questions

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    parser = argparse.ArgumentParser(
        description='Extract persona vectors from a language model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Extract silly vectors from Qwen:
    python extract_vectors.py --model qwen2.5-7b-instruct --trait silly

  Extract custom trait vectors:
    python extract_vectors.py --model llama-3.1-8b-instruct --trait confident --output vectors/

  Specify output directory:
    python extract_vectors.py --model mistral-7b-instruct-v0.3 --trait honest --output experiments/vectors/

Available models:
  - qwen2.5-7b-instruct
  - llama-3.1-8b-instruct
  - mistral-7b-instruct-v0.3
  - gpt2-medium
  - gpt2

Available traits:
  - silly (humorous vs serious)
  - honest (truthful vs deceptive)
  - Custom traits (if created via web interface)
        """
    )
    
    parser.add_argument(
        '--model',
        required=True,
        help='Model ID to extract vectors from'
    )
    
    parser.add_argument(
        '--trait',
        required=True,
        help='Personality trait to extract (e.g., silly, honest, confident)'
    )
    
    parser.add_argument(
        '--output',
        default='src/data/vectors',
        help='Output directory for vector files (default: src/data/vectors)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get trait prompts and questions
    logger.info(f"Loading trait definition: {args.trait}")
    prompt_pairs = get_prompt_pairs(args.trait)
    questions = get_evaluation_questions(args.trait)
    
    if not prompt_pairs:
        logger.error(f"Trait '{args.trait}' not found. Available: silly, honest, or custom traits.")
        sys.exit(1)
    
    if not questions:
        logger.error(f"No evaluation questions found for trait '{args.trait}'")
        sys.exit(1)
    
    logger.info(f"Trait: {args.trait}")
    logger.info(f"Prompt pairs: {len(prompt_pairs)}")
    logger.info(f"Evaluation questions: {len(questions)}")
    
    # Load model
    logger.info(f"Loading model: {args.model}")
    try:
        model, tokenizer = _load_model(args.model)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)
    
    # Extract vectors
    logger.info("Extracting persona vectors...")
    logger.info("This will take 3-5 minutes depending on model size")
    
    try:
        result = await generate_persona_vectors(
            model_id=args.model,
            trait_id=args.trait,
            prompt_pairs=prompt_pairs,
            questions=questions
        )
        
        # Save results
        output_file = output_dir / f"{args.model}_{args.trait}.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, cls=NumpyEncoder, indent=2)
        
        logger.info(f"\nExtraction complete!")
        logger.info(f"Vectors saved to: {output_file}")
        logger.info(f"Layers: {len(result['vectors'])}")
        
        # Show effectiveness scores
        if 'layer_scores' in result:
            logger.info("\nTop 5 most effective layers:")
            sorted_layers = sorted(
                result['layer_scores'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            for layer_name, score in sorted_layers:
                logger.info(f"  {layer_name}: {score:.4f}")
        
    except Exception as e:
        logger.error(f"Vector extraction failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
