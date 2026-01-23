#!/usr/bin/env python3
"""
Cross-Model Persona Transfer Benchmark Runner

Systematically evaluates persona vector transfer across:
- Multiple models (Qwen2.5-7B, LLaMA-3.1-8B, Mistral-7B)
- Multiple traits (silly, honest, helpful, rude, etc.)
- Multiple layers (early, mid, late)
- Multiple magnitudes (0, 0.5, 1, 2, 4)

Computes metrics:
- Trait strength (LLM judge)
- Coherence (perplexity)
"""

import argparse
import json
import logging
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from prompts import get_prompt_pairs, get_evaluation_questions
from models import load_model, apply_persona_steering

# Import BenchmarkEvaluator from evaluate_transfer
import importlib.util
spec = importlib.util.spec_from_file_location("evaluate_transfer", str(Path(__file__).parent / "evaluate_transfer.py"))
eval_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eval_module)
BenchmarkEvaluator = eval_module.BenchmarkEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Main runner for the cross-model persona transfer benchmark."""
    
    def __init__(self, config_path: str = "src/config/config.json"):
        """
        Initialize the benchmark runner.
        
        Args:
            config_path: Path to the benchmark configuration file (default: src/config/config.json)
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # Create output directory
        self.output_dir = Path(self.config['experiment_settings']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize results storage
        self.results = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'config_path': str(config_path),
                'config': self.config
            },
            'experiments': []
        }
        
        logger.info(f"Benchmark runner initialized. Output: {self.output_dir}")
    
    def _load_config(self) -> Dict:
        """Load benchmark configuration from JSON."""
        with open(self.config_path, 'r') as f:
            return json.load(f)
    
    def get_layer_indices(self, model_name: str, layer_type: str) -> List[int]:
        """
        Get layer indices for a specific layer type (early/mid/late).
        
        Args:
            model_name: Model identifier
            layer_type: One of 'early', 'mid', 'late', 'all'
        
        Returns:
            List of layer indices
        """
        model_config = self.config['models'][model_name]
        num_layers = model_config['num_layers']
        
        if layer_type == 'all':
            return list(range(num_layers))
        
        layer_config = self.config['layers'][layer_type]
        start_ratio, end_ratio = layer_config['range_ratio']
        
        start_idx = int(num_layers * start_ratio)
        end_idx = int(num_layers * end_ratio)
        
        return list(range(start_idx, end_idx))
    
    def run_single_experiment(
        self,
        source_model: str,
        target_model: str,
        trait_id: str,
        layer_type: str,
        alpha: float,
        num_eval_questions: int = 5
    ) -> Dict:
        """
        Run a single benchmark experiment.
        
        Args:
            source_model: Source model for vector extraction
            target_model: Target model for steering application
            trait_id: Trait identifier
            layer_type: Layer type (early/mid/late/all)
            alpha: Steering coefficient
            num_eval_questions: Number of evaluation questions to use
        
        Returns:
            Results dictionary with trait_strength and coherence metrics
        """
        experiment_id = f"{source_model}_{target_model}_{trait_id}_{layer_type}_a{alpha}"
        logger.info(f"Running experiment: {experiment_id}")
        
        try:
            # Get trait information
            trait_config = self.config['traits'][trait_id]
            
            # Load evaluation questions
            eval_questions = get_evaluation_questions(trait_id)
            eval_questions = eval_questions[:num_eval_questions]
            
            # Load/extract persona vectors
            vectors_path = Path(f"src/data/vectors/{source_model}_{trait_id}.json")
            if not vectors_path.exists():
                logger.warning(f"Vectors not found at {vectors_path}. Skipping.")
                return None
            
            with open(vectors_path, 'r') as f:
                vectors_data = json.load(f)
            
            # Get layer indices
            layer_indices = self.get_layer_indices(target_model, layer_type)
            
            # Load target model
            logger.info(f"Loading target model: {target_model}")
            model, tokenizer = load_model(target_model)
            
            # Initialize evaluator
            evaluator = BenchmarkEvaluator(
                target_model_name=self.config['models'][target_model]['hf_model_id'],
                judge_model_name=self.config['evaluation'].get('llm_judge_model', 'gpt-4-mini'),
                config_path=str(self.config_path)
            )
            
            # Generate responses and evaluate
            responses = []
            eval_results = []
            
            for question in eval_questions:
                try:
                    # Apply steering (simplified - just applies vector at specified layers)
                    response = apply_persona_steering(
                        model=model,
                        tokenizer=tokenizer,
                        prompt=question,
                        vectors=vectors_data,
                        layer_indices=layer_indices,
                        coefficient=alpha
                    )
                    
                    responses.append(response)
                    
                    # Evaluate response
                    eval_result = evaluator.evaluate_response(
                        response=response,
                        trait_id=trait_id,
                        trait_name=trait_config['name'],
                        trait_description=trait_config['description']
                    )
                    eval_results.append(eval_result)
                
                except Exception as e:
                    logger.error(f"Error generating response for question: {e}")
                    continue
            
            # Aggregate results
            agg_results = evaluator.aggregate_results(eval_results)
            
            return {
                'experiment_id': experiment_id,
                'source_model': source_model,
                'target_model': target_model,
                'trait_id': trait_id,
                'trait_name': trait_config['name'],
                'layer_type': layer_type,
                'layer_indices': layer_indices,
                'alpha': alpha,
                'num_questions': len(eval_questions),
                'num_responses': len(responses),
                'metrics': agg_results,
                'responses': responses if self.config['experiment_settings']['save_responses'] else [],
                'status': 'completed'
            }
        
        except Exception as e:
            logger.error(f"Experiment {experiment_id} failed: {e}")
            return {
                'experiment_id': experiment_id,
                'status': 'failed',
                'error': str(e)
            }
    
    def run_benchmark(
        self,
        source_models: Optional[List[str]] = None,
        target_models: Optional[List[str]] = None,
        traits: Optional[List[str]] = None,
        layer_types: Optional[List[str]] = None,
        alphas: Optional[List[float]] = None,
    ):
        """
        Run the full benchmark with specified parameters.
        
        Args:
            source_models: Models to extract vectors from (default: all in config)
            target_models: Models to apply steering to (default: all in config)
            traits: Traits to evaluate (default: all in config)
            layer_types: Layer types to test (default: all in config)
            alphas: Steering coefficients to test (default: config values)
        """
        # Use defaults from config if not specified
        source_models = source_models or list(self.config['models'].keys())
        target_models = target_models or list(self.config['models'].keys())
        traits = traits or list(self.config['traits'].keys())
        layer_types = layer_types or ['early', 'mid', 'late']
        alphas = alphas or self.config['magnitudes']['alpha_grid']
        
        # Total experiments
        total_exps = len(source_models) * len(target_models) * len(traits) * len(layer_types) * len(alphas)
        logger.info(f"Starting benchmark with {total_exps} total experiments")
        
        completed = 0
        
        # Run all experiment combinations
        for source_model in source_models:
            for target_model in target_models:
                for trait in traits:
                    for layer_type in layer_types:
                        for alpha in alphas:
                            result = self.run_single_experiment(
                                source_model=source_model,
                                target_model=target_model,
                                trait_id=trait,
                                layer_type=layer_type,
                                alpha=alpha
                            )
                            
                            if result:
                                self.results['experiments'].append(result)
                            
                            completed += 1
                            logger.info(f"Progress: {completed}/{total_exps}")
        
        logger.info(f"Benchmark complete. {len(self.results['experiments'])} experiments completed.")
    
    def save_results(self, filename: Optional[str] = None):
        """
        Save benchmark results to JSON file.
        
        Args:
            filename: Output filename (default: benchmark_results_{timestamp}.json)
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_{timestamp}.json"
        
        output_path = self.output_dir / filename
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")
        return output_path
    
    def generate_report(self) -> str:
        """
        Generate a human-readable summary report of results.
        
        Returns:
            Report string
        """
        report = []
        report.append("=" * 80)
        report.append("CROSS-MODEL PERSONA TRANSFER BENCHMARK REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {self.results['metadata']['timestamp']}")
        report.append("")
        
        # Summary statistics
        total_exps = len(self.results['experiments'])
        completed_exps = sum(1 for e in self.results['experiments'] if e['status'] == 'completed')
        failed_exps = sum(1 for e in self.results['experiments'] if e['status'] == 'failed')
        
        report.append(f"Total Experiments: {total_exps}")
        report.append(f"Completed: {completed_exps}")
        report.append(f"Failed: {failed_exps}")
        report.append("")
        
        # Results by trait
        report.append("RESULTS BY TRAIT:")
        report.append("-" * 80)
        
        by_trait = {}
        for exp in self.results['experiments']:
            if exp['status'] != 'completed':
                continue
            
            trait = exp['trait_id']
            if trait not in by_trait:
                by_trait[trait] = []
            
            by_trait[trait].append(exp)
        
        for trait in sorted(by_trait.keys()):
            exps = by_trait[trait]
            report.append(f"\n{trait}:")
            
            # Average metrics across all configurations
            trait_strengths = [e['metrics'].get('trait_strength_mean') for e in exps if 'trait_strength_mean' in e['metrics']]
            coherences = [e['metrics'].get('coherence_mean') for e in exps if 'coherence_mean' in e['metrics']]
            
            if trait_strengths:
                report.append(f"  Avg Trait Strength: {np.mean(trait_strengths):.2f} ± {np.std(trait_strengths):.2f}")
            if coherences:
                report.append(f"  Avg Coherence (perplexity): {np.mean(coherences):.2f} ± {np.std(coherences):.2f}")
        
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)
    
    def print_report(self):
        """Print the summary report to stdout."""
        report = self.generate_report()
        print(report)
        
        # Also save to file
        report_path = self.output_dir / "benchmark_report.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        logger.info(f"Report saved to {report_path}")


def main():
    """Main entry point for the benchmark runner."""
    parser = argparse.ArgumentParser(
        description="Cross-Model Persona Transfer Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full benchmark
  python benchmark_runner.py --config src/config/config.json
  
  # Run specific models and traits
  python benchmark_runner.py --config src/config/config.json \
    --source-models qwen2.5-7b-instruct llama-3.1-8b-instruct \
    --target-models qwen2.5-7b-instruct \
    --traits silly honest helpful \
    --alphas 0 0.5 1.0 2.0
  
  # Run quick test with limited scope
  python benchmark_runner.py --config src/config/config.json \
    --traits silly honest \
    --layer-types early late \
    --alphas 0 1.0
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='src/config/config.json',
        help='Path to benchmark configuration file'
    )
    parser.add_argument(
        '--source-models',
        nargs='+',
        help='Source models for vector extraction'
    )
    parser.add_argument(
        '--target-models',
        nargs='+',
        help='Target models for steering application'
    )
    parser.add_argument(
        '--traits',
        nargs='+',
        help='Traits to evaluate'
    )
    parser.add_argument(
        '--layer-types',
        nargs='+',
        choices=['early', 'mid', 'late', 'all'],
        help='Layer types to test'
    )
    parser.add_argument(
        '--alphas',
        nargs='+',
        type=float,
        help='Steering coefficients to test'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output filename for results (default: auto-generated)'
    )
    parser.add_argument(
        '--report-only',
        action='store_true',
        help='Print report from last results without running experiments'
    )
    
    args = parser.parse_args()
    
    # Initialize runner
    runner = BenchmarkRunner(config_path=args.config)
    
    # Run benchmark (or just print report)
    if not args.report_only:
        runner.run_benchmark(
            source_models=args.source_models,
            target_models=args.target_models,
            traits=args.traits,
            layer_types=args.layer_types,
            alphas=args.alphas
        )
        
        # Save results
        runner.save_results(filename=args.output)
    
    # Print report
    runner.print_report()


if __name__ == '__main__':
    main()
