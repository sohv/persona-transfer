"""
Main Experiment Runner - Persona Transfer Between Models

This script orchestrates the complete experiment:
1. Load source (Qwen) and target (LLaMA) models
2. Collect paired activations on WikiText
3. Learn ridge mapping via cross-validation
4. Extract persona vectors from both models
5. Apply all baseline methods
6. Generate text under all conditions
7. Evaluate and save results

Usage:
    python run_experiment.py --config config.json
"""

import torch
import argparse
import json
from pathlib import Path
from typing import Dict, List
import time
from datetime import datetime

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from activation_extractor import ActivationExtractor
from ridge_mapping import RidgeMapping, create_random_baseline
from baseline_methods import BaselineTransfer, SteeringConfig
from steered_generation import SteeredGenerator
from evaluation import Evaluator


class PersonaTransferExperiment:
    """
    Main experiment class orchestrating the full pipeline.
    """

    def __init__(self, config: dict):
        """
        Args:
            config: Configuration dictionary with experiment parameters
        """
        self.config = config
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

        # Paths
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Model configs
        self.source_model_name = config['source_model']  # e.g., "Qwen/Qwen-7B"
        self.target_model_name = config['target_model']  # e.g., "meta-llama/Llama-2-7b-hf"
        self.layer_idx = config['layer_idx']  # e.g., 16

        # Data configs
        self.n_training_samples = config.get('n_training_samples', 5000)
        self.n_test_prompts = config.get('n_test_prompts', 30)
        self.n_seeds = config.get('n_seeds', 3)

        # Steering configs
        self.alpha_values = config.get('alpha_values', [0.5, 1.0, 2.0, 3.0, 5.0])

        # Models (loaded lazily)
        self.source_model = None
        self.target_model = None
        self.source_tokenizer = None
        self.target_tokenizer = None

        # Extractors
        self.source_extractor = None
        self.target_extractor = None

        # Mapping
        self.ridge_mapping = None
        self.baseline_transfer = None

        print(f"✓ Initialized experiment")
        print(f"  Source: {self.source_model_name}")
        print(f"  Target: {self.target_model_name}")
        print(f"  Layer: {self.layer_idx}")
        print(f"  Output: {self.output_dir}")

    def load_models(self):
        """Load source and target models."""
        print("\n=== Loading Models ===")

        # Load source model (Qwen)
        print(f"Loading source model: {self.source_model_name}")
        self.source_model = AutoModelForCausalLM.from_pretrained(
            self.source_model_name,
            torch_dtype=torch.float16,
            device_map=self.device
        )
        self.source_tokenizer = AutoTokenizer.from_pretrained(self.source_model_name)

        # Load target model (LLaMA)
        print(f"Loading target model: {self.target_model_name}")
        self.target_model = AutoModelForCausalLM.from_pretrained(
            self.target_model_name,
            torch_dtype=torch.float16,
            device_map=self.device
        )
        self.target_tokenizer = AutoTokenizer.from_pretrained(self.target_model_name)

        # Create extractors
        self.source_extractor = ActivationExtractor(
            self.source_model,
            self.source_tokenizer,
            self.layer_idx,
            self.device
        )

        self.target_extractor = ActivationExtractor(
            self.target_model,
            self.target_tokenizer,
            self.layer_idx,
            self.device
        )

        # Get dimensions
        d_source = self.source_extractor.get_hidden_dim()
        d_target = self.target_extractor.get_hidden_dim()

        print(f"✓ Models loaded")
        print(f"  Source dim: {d_source}")
        print(f"  Target dim: {d_target}")

        # Initialize baseline transfer
        self.baseline_transfer = BaselineTransfer(d_source, d_target)

    def collect_training_data(self) -> tuple:
        """
        Collect paired activations on WikiText for learning mapping.

        Returns:
            (H_source, H_target) tensors
        """
        print(f"\n=== Collecting Training Data ===")
        print(f"Target: {self.n_training_samples} samples")

        # Load WikiText-103
        dataset = load_dataset("wikitext", "wikitext-103-v1", split="train")

        # Filter out short texts
        texts = []
        for item in dataset:
            text = item['text'].strip()
            if len(text) > 50:  # Skip very short samples
                texts.append(text)
            if len(texts) >= self.n_training_samples:
                break

        print(f"✓ Loaded {len(texts)} texts from WikiText")

        # Extract activations from source model
        print("Extracting source activations...")
        H_source = self.source_extractor.extract_batch(texts, position='last')

        # Extract activations from target model
        print("Extracting target activations...")
        H_target = self.target_extractor.extract_batch(texts, position='last')

        print(f"✓ Collected paired activations")
        print(f"  H_source: {H_source.shape}")
        print(f"  H_target: {H_target.shape}")

        # Save for later analysis
        torch.save({
            'H_source': H_source,
            'H_target': H_target,
            'texts': texts[:100]  # Save first 100 for reference
        }, self.output_dir / 'training_data.pt')

        return H_source, H_target

    def learn_mapping(self, H_source: torch.Tensor, H_target: torch.Tensor):
        """
        Learn ridge regression mapping via cross-validation.
        """
        print(f"\n=== Learning Mapping ===")

        d_source = H_source.shape[1]
        d_target = H_target.shape[1]

        self.ridge_mapping = RidgeMapping(d_source, d_target)

        # Cross-validate lambda
        lambda_values = self.config.get('lambda_values', [0.01, 0.1, 1.0, 10.0, 100.0])

        best_lambda, cv_errors = self.ridge_mapping.cross_validate(
            H_source,
            H_target,
            lambda_values=lambda_values,
            n_folds=5
        )

        # Analyze mapping
        metrics = self.ridge_mapping.analyze_mapping(save_dir=self.output_dir)

        # Save mapping
        self.ridge_mapping.save(self.output_dir / 'ridge_mapping.pt')

        # Save CV results
        with open(self.output_dir / 'cv_results.json', 'w') as f:
            json.dump({
                'lambda_values': lambda_values,
                'cv_errors': cv_errors,
                'best_lambda': best_lambda,
                'mapping_metrics': metrics
            }, f, indent=2)

        print(f"✓ Mapping learned and saved")

    def extract_persona_vectors(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Extract persona vectors from both models.

        Returns:
            Dictionary: persona -> {source_vector, target_vector_native, target_vector_mapped, ...}
        """
        print(f"\n=== Extracting Persona Vectors ===")

        # Load persona datasets from config
        persona_datasets = self.config['persona_datasets']

        persona_vectors = {}

        for persona_name, dataset_info in persona_datasets.items():
            print(f"\nProcessing persona: {persona_name}")

            positive_texts = dataset_info['positive_texts']
            negative_texts = dataset_info['negative_texts']

            # Extract from source model (Qwen)
            v_source = self.source_extractor.extract_persona_vector(
                positive_texts,
                negative_texts,
                position='last'
            )

            # Extract from target model (LLaMA) - this is the oracle
            v_target_native = self.target_extractor.extract_persona_vector(
                positive_texts,
                negative_texts,
                position='last'
            )

            # Apply all transfer methods
            transferred_vectors = self.baseline_transfer.apply_all_baselines(
                v_source,
                v_oracle=v_target_native,
                ridge_mapping=self.ridge_mapping,
                normalize=True
            )

            # Store all vectors
            persona_vectors[persona_name] = {
                'source': v_source,
                'oracle_native': v_target_native,
                **transferred_vectors
            }

            # Compute similarities to oracle
            similarities = self.baseline_transfer.compare_to_oracle(
                v_source,
                v_target_native,
                ridge_mapping=self.ridge_mapping
            )

            print(f"  Similarities to oracle:")
            for method, sim in sorted(similarities.items(), key=lambda x: x[1], reverse=True):
                print(f"    {method:15s}: {sim:.3f}")

        # Save persona vectors
        torch.save(persona_vectors, self.output_dir / 'persona_vectors.pt')

        return persona_vectors

    def run_generation_experiments(
        self,
        persona_vectors: Dict[str, Dict[str, torch.Tensor]]
    ) -> List[Dict]:
        """
        Generate text under all conditions.

        Returns:
            List of generation results
        """
        print(f"\n=== Running Generation Experiments ===")

        # Load test prompts
        test_prompts = self.config['test_prompts'][:self.n_test_prompts]

        # Conditions to test
        conditions = ['baseline', 'zero_pad', 'interpolate', 'random_proj', 'ridge_mapped', 'oracle_native']

        # Create generator
        generator = SteeredGenerator(
            self.target_model,
            self.target_tokenizer,
            device=self.device
        )

        all_results = []

        total_generations = len(test_prompts) * len(persona_vectors) * len(conditions) * self.n_seeds
        current = 0

        print(f"Total generations: {total_generations}")

        for persona_name, vectors in persona_vectors.items():
            print(f"\n--- Persona: {persona_name} ---")

            for condition in conditions:
                print(f"  Condition: {condition}")

                # Get steering vector for this condition
                if condition == 'baseline':
                    steering_vector = None
                else:
                    steering_vector = vectors[condition]

                for alpha in self.alpha_values:
                    for seed in range(self.n_seeds):
                        # Set seed for reproducibility
                        torch.manual_seed(seed)

                        for prompt in test_prompts:
                            current += 1

                            # Register steering (if not baseline)
                            if steering_vector is not None:
                                generator.clear_steering()
                                generator.register_steering(
                                    layer_idx=self.layer_idx,
                                    steering_vector=steering_vector,
                                    alpha=alpha,
                                    normalize=False  # Already normalized
                                )

                            # Generate
                            start_time = time.time()
                            output = generator.generate(
                                prompt,
                                max_new_tokens=self.config.get('max_new_tokens', 100),
                                temperature=self.config.get('temperature', 0.8),
                                do_sample=True
                            )
                            gen_time = time.time() - start_time

                            # Store result
                            all_results.append({
                                'persona': persona_name,
                                'condition': condition,
                                'alpha': alpha,
                                'seed': seed,
                                'prompt': prompt,
                                'output': output,
                                'generation_time': gen_time
                            })

                            if current % 10 == 0:
                                print(f"    Progress: {current}/{total_generations}")

        # Save raw results
        with open(self.output_dir / 'generation_results.json', 'w') as f:
            json.dump(all_results, f, indent=2)

        print(f"\n✓ Generated {len(all_results)} outputs")

        return all_results

    def evaluate_results(self, generation_results: List[Dict]):
        """
        Evaluate all generated outputs.
        """
        print(f"\n=== Evaluating Results ===")

        # Create evaluator (without GPT-4 for now - expensive)
        use_gpt4 = self.config.get('use_gpt4', False)
        evaluator = Evaluator(
            self.target_model,
            self.target_tokenizer,
            device=self.device,
            use_gpt4=use_gpt4
        )

        # Prepare trait descriptions
        trait_descriptions = {
            name: info.get('description', '')
            for name, info in self.config['persona_datasets'].items()
        }

        # Evaluate
        eval_results = evaluator.evaluate_batch(
            generation_results,
            trait_descriptions,
            compute_perplexity=True
        )

        # Save detailed results
        evaluator.save_results(eval_results, self.output_dir / 'eval_results.json')

        # Aggregate results
        aggregated = evaluator.aggregate_results(eval_results)

        # Print summary
        print("\n=== Results Summary ===")
        for condition, metrics in aggregated.items():
            print(f"\n{condition}:")
            for metric, value in metrics.items():
                print(f"  {metric:20s}: {value:.3f}")

        # Save aggregated results
        with open(self.output_dir / 'aggregated_results.json', 'w') as f:
            json.dump(aggregated, f, indent=2)

        return eval_results, aggregated

    def run(self):
        """Run the complete experiment pipeline."""
        start_time = time.time()

        print(f"\n{'='*60}")
        print(f"Starting Persona Transfer Experiment")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")

        # 1. Load models
        self.load_models()

        # 2. Collect training data
        H_source, H_target = self.collect_training_data()

        # 3. Learn mapping
        self.learn_mapping(H_source, H_target)

        # 4. Extract persona vectors
        persona_vectors = self.extract_persona_vectors()

        # 5. Run generation experiments
        generation_results = self.run_generation_experiments(persona_vectors)

        # 6. Evaluate results
        eval_results, aggregated = self.evaluate_results(generation_results)

        # Save config
        with open(self.output_dir / 'config.json', 'w') as f:
            json.dump(self.config, f, indent=2)

        total_time = time.time() - start_time

        print(f"\n{'='*60}")
        print(f"Experiment Complete!")
        print(f"Total time: {total_time/3600:.2f} hours")
        print(f"Results saved to: {self.output_dir}")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Run persona transfer experiment")
    parser.add_argument('--config', type=str, required=True, help="Path to config JSON")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = json.load(f)

    # Run experiment
    experiment = PersonaTransferExperiment(config)
    experiment.run()


if __name__ == "__main__":
    main()
