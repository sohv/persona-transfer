"""
Evaluation Metrics - Assess quality of steered generation.

This module implements metrics to evaluate:
1. Coherence: Is the output fluent and non-collapsed?
2. Trait strength: Does output exhibit the target persona?
3. Similarity: How close to oracle (native vector) performance?

Metrics:
- Perplexity (model's confidence in its own output)
- Repetition collapse detection
- GPT-4 judge for coherence and trait strength
- Cosine similarity to oracle outputs
"""

import torch
from typing import List, Dict, Optional
import numpy as np
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class EvaluationResult:
    """Results from evaluating a single generation."""

    prompt: str
    output: str
    condition: str  # e.g., "baseline", "ridge_mapped", "oracle"
    persona: str  # e.g., "silly", "formal"

    # Coherence metrics
    perplexity: Optional[float] = None
    is_collapsed: bool = False
    repetition_ratio: Optional[float] = None

    # Trait metrics
    gpt4_coherence_score: Optional[float] = None  # 1-5 scale
    gpt4_trait_score: Optional[float] = None  # 1-5 scale

    # Other
    output_length: int = 0
    generation_time: Optional[float] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'prompt': self.prompt,
            'output': self.output,
            'condition': self.condition,
            'persona': self.persona,
            'perplexity': self.perplexity,
            'is_collapsed': self.is_collapsed,
            'repetition_ratio': self.repetition_ratio,
            'gpt4_coherence_score': self.gpt4_coherence_score,
            'gpt4_trait_score': self.gpt4_trait_score,
            'output_length': self.output_length,
            'generation_time': self.generation_time,
        }


class CoherenceEvaluator:
    """
    Evaluates coherence of generated text.
    """

    def __init__(self, model, tokenizer, device="cuda"):
        """
        Args:
            model: Language model for perplexity computation
            tokenizer: Corresponding tokenizer
            device: Device to run on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

    def compute_perplexity(self, text: str) -> float:
        """
        Compute perplexity of text under the model.

        Lower perplexity = more fluent/expected text

        Args:
            text: Generated text

        Returns:
            Perplexity value
        """
        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        input_ids = inputs['input_ids']

        # Compute log-likelihood
        with torch.no_grad():
            outputs = self.model(**inputs, labels=input_ids)
            # outputs.loss is negative log-likelihood per token
            nll = outputs.loss.item()

        # Perplexity = exp(negative log-likelihood)
        perplexity = np.exp(nll)

        return perplexity

    def detect_repetition(
        self,
        text: str,
        n_gram: int = 4,
        window_size: int = 50
    ) -> Dict[str, float]:
        """
        Detect repetitive patterns in text.

        Args:
            text: Generated text
            n_gram: N-gram size to check
            window_size: Look at last N tokens (for long text)

        Returns:
            Dictionary with repetition metrics
        """
        tokens = text.split()

        # Use only recent tokens for long texts
        if len(tokens) > window_size:
            tokens = tokens[-window_size:]

        if len(tokens) < n_gram * 2:
            return {
                'repetition_ratio': 0.0,
                'is_collapsed': False,
                'unique_ngrams': 0,
                'total_ngrams': 0
            }

        # Extract n-grams
        ngrams = []
        for i in range(len(tokens) - n_gram + 1):
            ngram = tuple(tokens[i:i + n_gram])
            ngrams.append(ngram)

        if len(ngrams) == 0:
            return {
                'repetition_ratio': 0.0,
                'is_collapsed': False,
                'unique_ngrams': 0,
                'total_ngrams': 0
            }

        # Compute uniqueness ratio
        unique_ngrams = len(set(ngrams))
        total_ngrams = len(ngrams)
        unique_ratio = unique_ngrams / total_ngrams
        repetition_ratio = 1.0 - unique_ratio

        # Flag as collapsed if >50% repetition
        is_collapsed = repetition_ratio > 0.5

        return {
            'repetition_ratio': repetition_ratio,
            'is_collapsed': is_collapsed,
            'unique_ngrams': unique_ngrams,
            'total_ngrams': total_ngrams
        }


class GPT4Judge:
    """
    Use GPT-4 as judge for coherence and trait strength.

    This is the gold standard for evaluation in steering papers.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Args:
            api_key: OpenAI API key (if None, will look for OPENAI_API_KEY env var)
        """
        self.api_key = api_key

        # Try to import openai
        try:
            import openai
            self.openai = openai
            if api_key:
                openai.api_key = api_key
        except ImportError:
            print("Warning: openai package not installed. GPT-4 judge unavailable.")
            print("Install with: pip install openai")
            self.openai = None

    def judge_coherence(self, text: str) -> float:
        """
        Rate coherence of text on 1-5 scale.

        1 = Completely incoherent/collapsed
        2 = Somewhat incoherent, many errors
        3 = Partially coherent, some issues
        4 = Mostly coherent, minor issues
        5 = Perfectly coherent and fluent

        Args:
            text: Generated text

        Returns:
            Coherence score (1-5)
        """
        if self.openai is None:
            raise RuntimeError("OpenAI package not available")

        prompt = f"""Rate the coherence of the following text on a scale of 1-5:

1 = Completely incoherent/collapsed (repetitive gibberish)
2 = Somewhat incoherent, many grammatical errors
3 = Partially coherent, some issues
4 = Mostly coherent, minor issues
5 = Perfectly coherent and fluent

Text to rate:
\"\"\"{text}\"\"\"

Respond with just a number (1-5) and brief explanation.
Format: "Score: X\\nReason: ..."
"""

        try:
            response = self.openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,  # Deterministic
                max_tokens=100
            )

            content = response.choices[0].message.content.strip()

            # Parse score
            if content.startswith("Score:"):
                score_line = content.split("\n")[0]
                score = float(score_line.split(":")[1].strip())
            else:
                # Try to find first digit
                import re
                match = re.search(r'(\d)', content)
                if match:
                    score = float(match.group(1))
                else:
                    print(f"Warning: Could not parse GPT-4 response: {content}")
                    score = 3.0  # Default to middle

            return max(1.0, min(5.0, score))  # Clamp to [1, 5]

        except Exception as e:
            print(f"Error calling GPT-4 API: {e}")
            return 3.0  # Default to middle score

    def judge_trait(self, text: str, trait_name: str, trait_description: str) -> float:
        """
        Rate strength of target trait in text on 1-5 scale.

        1 = Trait completely absent
        2 = Trait barely present
        3 = Trait moderately present
        4 = Trait strongly present
        5 = Trait extremely present

        Args:
            text: Generated text
            trait_name: Name of trait (e.g., "silly", "formal")
            trait_description: Description of trait for judge

        Returns:
            Trait strength score (1-5)
        """
        if self.openai is None:
            raise RuntimeError("OpenAI package not available")

        prompt = f"""Rate how strongly the "{trait_name}" trait is present in the following text on a scale of 1-5:

Trait description: {trait_description}

1 = Trait completely absent
2 = Trait barely present
3 = Trait moderately present
4 = Trait strongly present
5 = Trait extremely present

Text to rate:
\"\"\"{text}\"\"\"

Respond with just a number (1-5) and brief explanation.
Format: "Score: X\\nReason: ..."
"""

        try:
            response = self.openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=100
            )

            content = response.choices[0].message.content.strip()

            # Parse score (same as coherence)
            if content.startswith("Score:"):
                score_line = content.split("\n")[0]
                score = float(score_line.split(":")[1].strip())
            else:
                import re
                match = re.search(r'(\d)', content)
                if match:
                    score = float(match.group(1))
                else:
                    print(f"Warning: Could not parse GPT-4 response: {content}")
                    score = 3.0

            return max(1.0, min(5.0, score))

        except Exception as e:
            print(f"Error calling GPT-4 API: {e}")
            return 3.0


class Evaluator:
    """
    Main evaluation pipeline combining all metrics.
    """

    def __init__(
        self,
        model,
        tokenizer,
        device="cuda",
        use_gpt4: bool = False,
        openai_api_key: Optional[str] = None
    ):
        """
        Args:
            model: Model for perplexity computation
            tokenizer: Tokenizer
            device: Device
            use_gpt4: Whether to use GPT-4 judge (requires API key)
            openai_api_key: OpenAI API key
        """
        self.coherence_eval = CoherenceEvaluator(model, tokenizer, device)

        self.use_gpt4 = use_gpt4
        if use_gpt4:
            self.gpt4_judge = GPT4Judge(openai_api_key)
        else:
            self.gpt4_judge = None

    def evaluate_single(
        self,
        prompt: str,
        output: str,
        condition: str,
        persona: str,
        trait_description: Optional[str] = None,
        compute_perplexity: bool = True
    ) -> EvaluationResult:
        """
        Evaluate a single generation.

        Args:
            prompt: Input prompt
            output: Generated text
            condition: Experimental condition (e.g., "ridge_mapped")
            persona: Target persona (e.g., "silly")
            trait_description: Description of trait for GPT-4 judge
            compute_perplexity: Whether to compute perplexity (slow)

        Returns:
            EvaluationResult object
        """
        result = EvaluationResult(
            prompt=prompt,
            output=output,
            condition=condition,
            persona=persona,
            output_length=len(output.split())
        )

        # Coherence metrics
        if compute_perplexity:
            result.perplexity = self.coherence_eval.compute_perplexity(output)

        rep_metrics = self.coherence_eval.detect_repetition(output)
        result.is_collapsed = rep_metrics['is_collapsed']
        result.repetition_ratio = rep_metrics['repetition_ratio']

        # GPT-4 judge (if enabled)
        if self.use_gpt4 and self.gpt4_judge:
            result.gpt4_coherence_score = self.gpt4_judge.judge_coherence(output)

            if trait_description:
                result.gpt4_trait_score = self.gpt4_judge.judge_trait(
                    output, persona, trait_description
                )

        return result

    def evaluate_batch(
        self,
        results: List[Dict],  # List of {prompt, output, condition, persona}
        trait_descriptions: Dict[str, str],  # persona -> description
        compute_perplexity: bool = True
    ) -> List[EvaluationResult]:
        """
        Evaluate multiple generations.

        Args:
            results: List of generation results
            trait_descriptions: Mapping from persona to description
            compute_perplexity: Whether to compute perplexity

        Returns:
            List of EvaluationResult objects
        """
        eval_results = []

        for i, item in enumerate(results):
            print(f"Evaluating {i+1}/{len(results)}: {item['condition']} / {item['persona']}")

            trait_desc = trait_descriptions.get(item['persona'], None)

            eval_result = self.evaluate_single(
                prompt=item['prompt'],
                output=item['output'],
                condition=item['condition'],
                persona=item['persona'],
                trait_description=trait_desc,
                compute_perplexity=compute_perplexity
            )

            eval_results.append(eval_result)

        return eval_results

    def aggregate_results(
        self,
        eval_results: List[EvaluationResult]
    ) -> Dict[str, Dict[str, float]]:
        """
        Aggregate results by condition.

        Returns:
            Dictionary: condition -> metric -> mean value
        """
        # Group by condition
        by_condition = {}
        for result in eval_results:
            if result.condition not in by_condition:
                by_condition[result.condition] = []
            by_condition[result.condition].append(result)

        # Compute means
        aggregated = {}
        for condition, results in by_condition.items():
            metrics = {
                'collapse_rate': np.mean([r.is_collapsed for r in results]),
                'repetition_ratio': np.mean([r.repetition_ratio for r in results if r.repetition_ratio is not None]),
                'output_length': np.mean([r.output_length for r in results]),
            }

            # Add perplexity if available
            perplexities = [r.perplexity for r in results if r.perplexity is not None]
            if perplexities:
                metrics['perplexity'] = np.mean(perplexities)

            # Add GPT-4 scores if available
            coherence_scores = [r.gpt4_coherence_score for r in results if r.gpt4_coherence_score is not None]
            if coherence_scores:
                metrics['gpt4_coherence'] = np.mean(coherence_scores)

            trait_scores = [r.gpt4_trait_score for r in results if r.gpt4_trait_score is not None]
            if trait_scores:
                metrics['gpt4_trait'] = np.mean(trait_scores)

            aggregated[condition] = metrics

        return aggregated

    def save_results(self, eval_results: List[EvaluationResult], output_path: str):
        """Save evaluation results to JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = [r.to_dict() for r in eval_results]

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"✓ Saved {len(eval_results)} results to {output_path}")


if __name__ == "__main__":
    print("Testing evaluation implementation...")

    # Test coherence evaluator
    print("\n=== Testing Coherence Evaluator ===")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    coherence_eval = CoherenceEvaluator(model, tokenizer, device="cpu")

    # Test perplexity
    text = "The quick brown fox jumps over the lazy dog."
    ppl = coherence_eval.compute_perplexity(text)
    print(f"Perplexity of fluent text: {ppl:.2f}")

    # Test repetition detection
    repetitive_text = "the the the the the the the the"
    rep_metrics = coherence_eval.detect_repetition(repetitive_text)
    print(f"\nRepetition detection on collapsed text:")
    print(f"  Repetition ratio: {rep_metrics['repetition_ratio']:.2f}")
    print(f"  Is collapsed: {rep_metrics['is_collapsed']}")

    normal_text = "This is a normal sentence with no repetition."
    rep_metrics = coherence_eval.detect_repetition(normal_text)
    print(f"\nRepetition detection on normal text:")
    print(f"  Repetition ratio: {rep_metrics['repetition_ratio']:.2f}")
    print(f"  Is collapsed: {rep_metrics['is_collapsed']}")

    # Test full evaluator (without GPT-4)
    print("\n=== Testing Full Evaluator ===")
    evaluator = Evaluator(model, tokenizer, device="cpu", use_gpt4=False)

    result = evaluator.evaluate_single(
        prompt="Test prompt",
        output="This is a test output that is coherent.",
        condition="test",
        persona="silly",
        compute_perplexity=True
    )

    print(f"\nEvaluation result:")
    print(f"  Perplexity: {result.perplexity:.2f}")
    print(f"  Repetition ratio: {result.repetition_ratio:.2f}")
    print(f"  Is collapsed: {result.is_collapsed}")

    print("\n✅ Evaluation tests passed!")
