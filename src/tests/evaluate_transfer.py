#!/usr/bin/env python3
"""
Evaluate cross-model persona transfer with systematic testing and metrics.

Integrated metrics:
- Trait Strength: LLM-based judge evaluation (0-10 scale)
- Coherence: Perplexity-based response quality assessment

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
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from models import _load_model, _generate_text_response, unload_model
from persona_vectors import load_persona_vectors, load_persona_vectors_from_file
from prompts import get_evaluation_questions

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TraitStrengthJudge:
    """Evaluate trait strength using LLM judge."""
    
    def __init__(self, judge_model_name: str = "gpt-4o-mini-2024-07-18"):
        self.judge_model_name = judge_model_name
        self.judge_prompt_template = """Rate how strongly the following trait is expressed in this response on a scale of 0–10.
Trait: {trait_name}
Description: {trait_description}
Response: "{response}"
Output only a single number from 0 to 10."""
        
        self._init_judge_model()
    
    def _init_judge_model(self):
        """Initialize the judge LLM."""
        if self.judge_model_name.startswith("gpt"):
            self.use_openai = True
            try:
                import openai
                self.client = openai.OpenAI()
                logger.info(f"Using OpenAI judge model: {self.judge_model_name}")
            except ImportError:
                logger.warning("OpenAI API not available. Install with: pip install openai")
                self.use_openai = False
        else:
            self.use_openai = False
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.judge_model_name)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.judge_model_name,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
                logger.info(f"Loaded local judge model: {self.judge_model_name}")
            except Exception as e:
                logger.warning(f"Could not load judge model {self.judge_model_name}: {e}")
                self.model = None
                self.tokenizer = None
    
    def evaluate(
        self,
        response: str,
        trait_name: str,
        trait_description: str
    ) -> Optional[float]:
        """Evaluate trait strength in response (0-10 scale)."""
        prompt = self.judge_prompt_template.format(
            trait_name=trait_name,
            trait_description=trait_description,
            response=response[:1000]  # Truncate to 1000 chars
        )
        
        try:
            if self.use_openai:
                msg = self.client.chat.completions.create(
                    model=self.judge_model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=10
                )
                score_text = msg.choices[0].message.content.strip()
            else:
                if self.model is None:
                    return None
                tokens = self.tokenizer.encode(prompt, return_tensors="pt")
                output = self.model.generate(tokens, max_new_tokens=5, temperature=0.0)
                score_text = self.tokenizer.decode(output[0]).split(prompt)[-1].strip()
            
            # Extract number from response
            score = float(''.join(c for c in score_text if c.isdigit() or c == '.'))
            return max(0.0, min(10.0, score))
        except Exception as e:
            logger.warning(f"Judge evaluation failed: {e}")
            return None


class CoherenceMetric:
    """Evaluate response coherence using perplexity."""
    
    def __init__(self, model_name: str = "gpt2"):
        """Initialize coherence metric with a language model."""
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load tokenizer and model for perplexity computation."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            self.model.eval()
            if torch.cuda.is_available():
                self.model.cuda()
        except Exception as e:
            logger.warning(f"Could not load coherence model {self.model_name}: {e}")
    
    def evaluate(self, response: str) -> float:
        """
        Compute perplexity-based coherence score (0-100).
        Higher score = better coherence.
        """
        if not response or len(response.strip()) < 5:
            return 0.0
        
        if self.model is None or self.tokenizer is None:
            return self.simple_coherence(response)
        
        try:
            encodings = self.tokenizer(response, return_tensors="pt")
            max_length = self.model.config.n_positions if hasattr(self.model.config, 'n_positions') else 512
            stride = 512
            
            nlls = []
            for i in range(0, encodings.input_ids.size(1), stride):
                begin_loc = max(0, i - max_length)
                end_loc = min(encodings.input_ids.size(1), i + stride)
                trg_len = end_loc - i
                input_ids = encodings.input_ids[:, begin_loc:end_loc]
                
                if torch.cuda.is_available():
                    input_ids = input_ids.cuda()
                
                with torch.no_grad():
                    outputs = self.model(input_ids, labels=input_ids)
                    neg_log_likelihood = outputs.loss
                    nlls.append(neg_log_likelihood)
            
            ppl = torch.exp(torch.stack(nlls).mean()).item()
            # Convert to 0-100 scale (lower perplexity = higher score)
            coherence = 100.0 / (1.0 + np.log(ppl + 1))
            return max(0.0, min(100.0, coherence))
        except Exception as e:
            logger.debug(f"Perplexity computation failed: {e}")
            return self.simple_coherence(response)
    
    @staticmethod
    def simple_coherence(text: str) -> float:
        """Simple coherence scoring based on text quality indicators."""
        if not text or len(text.strip()) < 10:
            return 0.0
        
        score = 50.0
        
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
        if text and text[0].isupper():
            score += 5
        
        # Penalize excessive repetition
        words = text.lower().split()
        if len(words) > 0:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.5:
                score -= 20
        
        return max(0.0, min(100.0, score))


class BenchmarkEvaluator:
    """
    Main evaluator for the cross-model persona transfer benchmark.
    
    Computes both trait strength and coherence metrics for responses.
    """
    
    def __init__(
        self,
        target_model_name: str,
        judge_model_name: str = "gpt-4o-mini-2024-07-18",
        config_path: Optional[str] = None
    ):
        """
        Initialize the benchmark evaluator.
        
        Args:
            target_model_name: Model to compute coherence under
            judge_model_name: Judge model for trait strength evaluation
            config_path: Optional path to benchmark config.json
        """
        self.target_model_name = target_model_name
        self.judge_model_name = judge_model_name
        
        # Load config if provided
        self.config = {}
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                self.config = json.load(f).get('traits', {})
        
        # Initialize metrics
        self.trait_judge = TraitStrengthJudge(judge_model_name)
        self.coherence_metric = CoherenceMetric(target_model_name)
    
    def evaluate_response(
        self,
        response: str,
        trait_id: str,
        trait_name: Optional[str] = None,
        trait_description: Optional[str] = None
    ) -> Dict[str, Optional[float]]:
        """
        Evaluate a response for trait strength and coherence.
        
        Args:
            response: The model response to evaluate
            trait_id: Identifier for the trait
            trait_name: Name of the trait (uses config if not provided)
            trait_description: Description of the trait (uses config if not provided)
        
        Returns:
            Dict with keys:
                - trait_strength: 0-10 score
                - coherence: perplexity score (lower is better)
        """
        # Use config values if not provided
        if trait_id in self.config:
            if trait_name is None:
                trait_name = self.config[trait_id].get('name', trait_id)
            if trait_description is None:
                trait_description = self.config[trait_id].get('description', trait_id)
        
        trait_name = trait_name or trait_id
        trait_description = trait_description or trait_id
        
        # Evaluate trait strength and coherence
        trait_strength = self.trait_judge.evaluate(response, trait_name, trait_description)
        coherence = self.coherence_metric.evaluate(response)
        
        return {
            "trait_strength": trait_strength,
            "coherence": coherence
        }
    
    def evaluate_batch(
        self,
        responses: List[str],
        trait_id: str,
        trait_name: Optional[str] = None,
        trait_description: Optional[str] = None
    ) -> List[Dict[str, Optional[float]]]:
        """
        Evaluate a batch of responses.
        
        Args:
            responses: List of responses to evaluate
            trait_id: Identifier for the trait
            trait_name: Name of the trait
            trait_description: Description of the trait
        
        Returns:
            List of evaluation dicts
        """
        results = []
        for response in responses:
            result = self.evaluate_response(response, trait_id, trait_name, trait_description)
            results.append(result)
        return results
    
    def aggregate_results(
        self,
        eval_results: List[Dict[str, Optional[float]]]
    ) -> Dict[str, float]:
        """
        Aggregate evaluation results across multiple responses.
        
        Args:
            eval_results: List of evaluation dicts from evaluate_batch
        
        Returns:
            Dict with aggregated metrics:
                - trait_strength_mean: Average trait strength (0-10)
                - trait_strength_std: Std dev of trait strength
                - coherence_mean: Average coherence
                - coherence_std: Std dev of coherence
        """
        trait_strengths = [r['trait_strength'] for r in eval_results if r['trait_strength'] is not None]
        coherences = [r['coherence'] for r in eval_results if r['coherence'] is not None]
        
        results = {}
        
        if trait_strengths:
            results['trait_strength_mean'] = np.mean(trait_strengths)
            results['trait_strength_std'] = np.std(trait_strengths)
        
        if coherences:
            results['coherence_mean'] = np.mean(coherences)
            results['coherence_std'] = np.std(coherences)
        
        return results


async def evaluate_single_transfer(
    source_model_id,
    target_model_id,
    trait,
    vector_file,
    prompts,
    coefficients,
    trait_description,
    max_tokens=150,
    use_trait_judge=True
):
    """Evaluate transfer for one source-target pair with integrated metrics."""
    
    logger.info(f"\nEvaluating: {source_model_id} → {target_model_id}")
    logger.info(f"Trait: {trait} ({trait_description})")
    logger.info(f"Coefficients: {coefficients}")
    logger.info(f"Test prompts: {len(prompts)}")
    
    # Load vectors: if a filepath was provided, use the synchronous file loader;
    # otherwise attempt to load by model/trait via the async loader.
    if isinstance(vector_file, (str, Path)):
        vector_path = Path(vector_file)
        vector_data = load_persona_vectors_from_file(vector_path)
    else:
        # Expecting (model_id, trait_id) style parameters for async loader
        vector_data = await load_persona_vectors(source_model_id, trait)

    if not vector_data:
        raise ValueError(f"Failed to load vectors from {vector_file}")
    
    # Load target model
    logger.info(f"Loading target model: {target_model_id}")
    model, tokenizer = _load_model(target_model_id)
    
    # Initialize metrics
    coherence_metric = CoherenceMetric()
    trait_judge = TraitStrengthJudge() if use_trait_judge else None
    
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
                    steering_coefficient=coefficient,
                    source_model_id=source_model_id  # Pass source model ID for dimension mapping
                )
                
                # Compute metrics
                coherence = coherence_metric.evaluate(response)
                trait_strength = None
                if trait_judge:
                    trait_strength = trait_judge.evaluate(response, trait, trait_description)
                
                prompt_results['responses'].append({
                    'coefficient': coefficient,
                    'response': response,
                    'coherence': coherence,
                    'trait_strength': trait_strength,
                    'length': len(response.split())
                })
                
                logger.debug(f"  Response length: {len(response.split())} words, "
                           f"Coherence: {coherence:.1f}, "
                           f"Trait Strength: {trait_strength if trait_strength else 'N/A'}")
                
            except Exception as e:
                logger.error(f"  Failed: {e}")
                prompt_results['responses'].append({
                    'coefficient': coefficient,
                    'response': None,
                    'error': str(e),
                    'coherence': 0.0,
                    'trait_strength': None
                })
        
        results.append(prompt_results)
    
    elapsed = time.time() - start_time
    logger.info(f"Evaluation completed in {elapsed:.1f}s")
    
    # Unload model to free memory
    unload_model(target_model_id)
    
    # Determine transfer type
    is_cross_model = source_model_id != target_model_id
    transfer_type = "cross-model" if is_cross_model else "same-model"

    # Check if dimension mapping was needed
    import json
    with open(vector_file, 'r') as f:
        vector_metadata = json.load(f)

    source_vectors = vector_metadata.get('vectors', {})
    if source_vectors:
        first_vector = next(iter(source_vectors.values()))
        source_dim = len(first_vector)
    else:
        source_dim = None

    # Get target dimension from config
    with open("src/config/config.json") as f:
        config = json.load(f)
    target_dim = config['models'].get(target_model_id, {}).get('hidden_size', source_dim)

    dimension_mapping_used = source_dim != target_dim if source_dim and target_dim else False

    return {
        'source_model': source_model_id,
        'target_model': target_model_id,
        'transfer_type': transfer_type,
        'dimension_mapping_used': dimension_mapping_used,
        'source_dimension': source_dim,
        'target_dimension': target_dim,
        'trait': trait,
        'trait_description': trait_description,
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
    
    # Load trait description from config
    import json
    with open("src/config/config.json") as f:
        config = json.load(f)
    
    if args.trait not in config['traits']:
        logger.error(f"Trait not found in config: {args.trait}")
        logger.info(f"Available traits: {', '.join(config['traits'].keys())}")
        sys.exit(1)
    
    trait_description = config['traits'][args.trait].get('description', args.trait)
    
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
    logger.info("CROSS-MODEL TRANSFER EVALUATION WITH INTEGRATED METRICS")
    logger.info("="*80)
    
    try:
        evaluation_results = asyncio.run(
            evaluate_single_transfer(
                source_model_id=args.source,
                target_model_id=args.target,
                trait=args.trait,
                trait_description=trait_description,
                vector_file=vector_file,
                prompts=prompts,
                coefficients=args.coefficients,
                max_tokens=args.max_tokens,
                use_trait_judge=True
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
        logger.info("SUMMARY - INTEGRATED METRICS")
        logger.info("="*80)
        
        avg_coherence_by_coef = {}
        avg_trait_strength_by_coef = {}
        
        for prompt_result in evaluation_results['results']:
            for response in prompt_result['responses']:
                coef = response['coefficient']
                coherence = response.get('coherence', 0.0)
                trait_strength = response.get('trait_strength', None)
                
                if coef not in avg_coherence_by_coef:
                    avg_coherence_by_coef[coef] = []
                    avg_trait_strength_by_coef[coef] = []
                
                avg_coherence_by_coef[coef].append(coherence)
                if trait_strength is not None:
                    avg_trait_strength_by_coef[coef].append(trait_strength)
        
        logger.info("\nCoherence (Response Quality) by Coefficient:")
        for coef in sorted(avg_coherence_by_coef.keys()):
            scores = avg_coherence_by_coef[coef]
            avg = sum(scores) / len(scores) if scores else 0.0
            logger.info(f"  {coef:+.1f}: {avg:.1f}/100")
        
        logger.info("\nTrait Strength (Persona Expression) by Coefficient:")
        for coef in sorted(avg_trait_strength_by_coef.keys()):
            scores = avg_trait_strength_by_coef[coef]
            if scores:
                avg = sum(scores) / len(scores)
                logger.info(f"  {coef:+.1f}: {avg:.1f}/10")
            else:
                logger.info(f"  {coef:+.1f}: N/A (LLM judge unavailable)")
        
        logger.info(f"\nTotal runtime: {evaluation_results['elapsed_seconds']:.1f}s")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
