"""
Steered Generation - Generate text with persona vector steering.

This module implements activation steering during text generation.
The key idea: add a persona vector to hidden states at a specific layer
during the forward pass, biasing the model's behavior.

Theory:
Given a persona vector v and steering strength α, we modify hidden states:
    h'_layer = h_layer + α * v

This is applied at each token generation step, continuously steering behavior.
"""

import torch
from typing import Optional, Callable, List
from transformers import PreTrainedModel, PreTrainedTokenizer
import numpy as np


class SteeringHook:
    """
    Hook that adds steering vector to hidden states during forward pass.
    """

    def __init__(
        self,
        layer_idx: int,
        steering_vector: torch.Tensor,
        alpha: float = 1.0,
        normalize: bool = True
    ):
        """
        Args:
            layer_idx: Which layer to apply steering
            steering_vector: Persona vector to add (d_model,)
            alpha: Steering strength coefficient
            normalize: Whether to normalize vector before applying
        """
        self.layer_idx = layer_idx
        self.alpha = alpha

        # Normalize steering vector if requested
        if normalize:
            self.steering_vector = steering_vector / (steering_vector.norm() + 1e-8)
        else:
            self.steering_vector = steering_vector

        # Move to same device as model (will be set when registered)
        self.device = None

        # Track if hook is active
        self.active = True

    def __call__(self, module, input_tuple, output):
        """
        Hook function called during forward pass.

        Args:
            module: The module this hook is attached to
            input_tuple: Inputs to the module (not used)
            output: Output from the module (hidden states)

        Returns:
            Modified output with steering applied
        """
        if not self.active:
            return output

        # Handle different output formats
        # Some models return tuple (hidden_states, ...), others just tensor
        if isinstance(output, tuple):
            hidden_states = output[0]  # (batch, seq, hidden_dim)
            other_outputs = output[1:]
        else:
            hidden_states = output
            other_outputs = None

        # Ensure steering vector is on correct device
        if self.device is None:
            self.device = hidden_states.device
            self.steering_vector = self.steering_vector.to(self.device)

        # Add steering vector: h' = h + α * v
        # Broadcasting: (batch, seq, d) + (d,) -> (batch, seq, d)
        steered_hidden = hidden_states + self.alpha * self.steering_vector

        # Return in same format as input
        if other_outputs is not None:
            return (steered_hidden,) + other_outputs
        else:
            return steered_hidden

    def deactivate(self):
        """Temporarily deactivate steering without removing hook."""
        self.active = False

    def activate(self):
        """Reactivate steering."""
        self.active = True


class SteeredGenerator:
    """
    Wrapper around HuggingFace model that enables steered generation.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: str = "cuda"
    ):
        """
        Args:
            model: HuggingFace causal LM
            tokenizer: Corresponding tokenizer
            device: Device to run on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        self.model.to(device)
        self.model.eval()

        # Track registered hooks
        self.hooks = []
        self.hook_handles = []

    def register_steering(
        self,
        layer_idx: int,
        steering_vector: torch.Tensor,
        alpha: float = 1.0,
        normalize: bool = True
    ) -> None:
        """
        Register steering vector at specified layer.

        Args:
            layer_idx: Which layer to steer
            steering_vector: Persona vector (d_model,)
            alpha: Steering strength
            normalize: Whether to normalize vector
        """
        # Create hook
        hook = SteeringHook(layer_idx, steering_vector, alpha, normalize)

        # Get the layer module
        # For most models: model.model.layers[i] or model.transformer.h[i]
        layer_module = self._get_layer_module(layer_idx)

        # Register hook
        handle = layer_module.register_forward_hook(hook)

        # Track for later removal
        self.hooks.append(hook)
        self.hook_handles.append(handle)

        print(f"✓ Registered steering at layer {layer_idx} with α={alpha:.2f}")

    def _get_layer_module(self, layer_idx: int):
        """
        Get the transformer layer module at specified index.

        Different models have different architecture:
        - LLaMA/Mistral: model.model.layers[i]
        - GPT-2: model.transformer.h[i]
        - Qwen: model.transformer.h[i] or model.model.layers[i]
        """
        # Try common patterns
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            # LLaMA-style
            return self.model.model.layers[layer_idx]
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            # GPT-2-style
            return self.model.transformer.h[layer_idx]
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'layers'):
            # Alternative transformer style
            return self.model.transformer.layers[layer_idx]
        else:
            raise ValueError(
                f"Cannot find layer module. Model type: {type(self.model)}. "
                f"Available attributes: {dir(self.model)}"
            )

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.9,
        do_sample: bool = True,
        **kwargs
    ) -> str:
        """
        Generate text with active steering.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            do_sample: Whether to sample (vs. greedy)
            **kwargs: Additional generation arguments

        Returns:
            Generated text (including prompt)
        """
        # Tokenize prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Generate with steering active
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )

        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return generated_text

    def generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 100,
        **kwargs
    ) -> List[str]:
        """
        Generate from multiple prompts in batch.

        Args:
            prompts: List of input prompts
            max_new_tokens: Max tokens per generation
            **kwargs: Additional generation arguments

        Returns:
            List of generated texts
        """
        # Tokenize batch
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )

        # Decode all
        generated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return generated_texts

    def clear_steering(self) -> None:
        """Remove all steering hooks."""
        for handle in self.hook_handles:
            handle.remove()

        self.hooks.clear()
        self.hook_handles.clear()

        print("✓ Cleared all steering hooks")

    def set_alpha(self, alpha: float) -> None:
        """Change steering strength for all registered hooks."""
        for hook in self.hooks:
            hook.alpha = alpha

        print(f"✓ Updated steering strength to α={alpha:.2f}")

    def deactivate_steering(self) -> None:
        """Temporarily disable steering without removing hooks."""
        for hook in self.hooks:
            hook.deactivate()

    def activate_steering(self) -> None:
        """Reactivate previously disabled steering."""
        for hook in self.hooks:
            hook.activate()


def detect_repetition_collapse(text: str, n_gram: int = 4, threshold: float = 0.5) -> bool:
    """
    Detect if generated text has collapsed into repetition.

    Args:
        text: Generated text
        n_gram: N-gram size to check
        threshold: Fraction of n-grams that can be repeated before flagging

    Returns:
        True if collapse detected
    """
    tokens = text.split()

    if len(tokens) < n_gram * 2:
        return False  # Too short to judge

    # Extract n-grams
    ngrams = []
    for i in range(len(tokens) - n_gram + 1):
        ngram = tuple(tokens[i:i + n_gram])
        ngrams.append(ngram)

    if len(ngrams) == 0:
        return False

    # Count unique n-grams
    unique_ratio = len(set(ngrams)) / len(ngrams)

    # If most n-grams are repeated, flag as collapse
    return unique_ratio < (1 - threshold)


def extract_completion_only(full_text: str, prompt: str) -> str:
    """
    Extract only the generated completion, removing the prompt.

    Args:
        full_text: Full generated text (prompt + completion)
        prompt: Original prompt

    Returns:
        Just the completion
    """
    if full_text.startswith(prompt):
        return full_text[len(prompt):].strip()
    else:
        # Fallback: return everything (prompt might be slightly modified by tokenizer)
        return full_text


if __name__ == "__main__":
    print("Testing steered generation implementation...")
    print("\nNote: This test requires loading a real model.")
    print("For quick testing, we'll use GPT-2 small.\n")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Load small model for testing
    model_name = "gpt2"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Create generator
    generator = SteeredGenerator(model, tokenizer, device="cpu")

    # Create a dummy steering vector
    hidden_dim = model.config.n_embd
    steering_vector = torch.randn(hidden_dim)

    # Register steering
    generator.register_steering(
        layer_idx=6,  # Middle layer
        steering_vector=steering_vector,
        alpha=0.5
    )

    # Generate
    prompt = "Once upon a time"
    print(f"Prompt: {prompt}")
    print("\nGenerating with steering...")

    output = generator.generate(
        prompt,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.8
    )

    print(f"\nOutput: {output}")

    # Test collapse detection
    repetitive_text = "the the the the the the the the"
    is_collapsed = detect_repetition_collapse(repetitive_text, n_gram=2, threshold=0.5)
    print(f"\nRepetition detection test: {is_collapsed} (should be True)")

    # Clean up
    generator.clear_steering()

    print("\n✅ Steered generation tests passed!")
