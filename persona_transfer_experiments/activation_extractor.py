"""
Activation Extractor - Consistent extraction protocol for all experiments.

CRITICAL: This class ensures consistency between:
1. Training data collection (for learning W)
2. Persona vector extraction from source model (Qwen)
3. Persona vector extraction from target model (LLaMA, oracle)
4. Steering during generation

All extraction uses the SAME protocol: last token position at specified layer.
"""

import torch
from typing import List, Literal
from transformers import PreTrainedModel
import numpy as np


class ActivationExtractor:
    """
    Extracts hidden state activations consistently across all experiments.

    Design principle: The extraction method must match the steering application.
    Since steering adds vectors to the last token before generation, we extract
    at the last token position.
    """

    def __init__(self, model: PreTrainedModel, tokenizer, layer_idx: int, device: str = "cuda"):
        """
        Args:
            model: HuggingFace transformer model
            tokenizer: Corresponding tokenizer
            layer_idx: Which layer to extract from (0-indexed)
            device: cuda or cpu
        """
        self.model = model
        self.tokenizer = tokenizer
        self.layer_idx = layer_idx
        self.device = device
        self.model.to(device)
        self.model.eval()

    def extract(self, text: str, position: Literal['last', 'mean'] = 'last') -> torch.Tensor:
        """
        Extract activation from a single text.

        Args:
            text: Input text string
            position: 'last' (default) or 'mean' (for ablation only)

        Returns:
            Tensor of shape (hidden_dim,)
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=512  # Reasonable limit
        ).to(self.device)

        # Forward pass with hidden states
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True
            )

        # Extract from specified layer
        # outputs.hidden_states is tuple of (num_layers + 1) tensors
        # Each tensor is (batch_size, seq_len, hidden_dim)
        hidden = outputs.hidden_states[self.layer_idx]  # (1, seq_len, d)

        if position == 'last':
            # Last token position - this is where steering happens during generation
            activation = hidden[:, -1, :].squeeze(0)  # (d,)
        elif position == 'mean':
            # Mean pooling across sequence (ablation only)
            activation = hidden.mean(dim=1).squeeze(0)  # (d,)
        else:
            raise ValueError(f"Unknown position: {position}")

        return activation.cpu()  # Move to CPU to save GPU memory

    def extract_batch(self, texts: List[str], position: Literal['last', 'mean'] = 'last') -> torch.Tensor:
        """
        Extract activations from multiple texts (for training data collection).

        Args:
            texts: List of input strings
            position: 'last' or 'mean'

        Returns:
            Tensor of shape (num_texts, hidden_dim)
        """
        activations = []

        # Process in batches to avoid OOM
        batch_size = 8
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]

            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,  # Pad to longest in batch
                truncation=True,
                max_length=512
            ).to(self.device)

            # Forward pass
            with torch.no_grad():
                outputs = self.model(
                    **inputs,
                    output_hidden_states=True,
                    return_dict=True
                )

            hidden = outputs.hidden_states[self.layer_idx]  # (batch, seq_len, d)

            if position == 'last':
                # Get last token for each sequence
                # Need to account for padding - use attention mask
                attention_mask = inputs['attention_mask']  # (batch, seq_len)
                last_token_indices = attention_mask.sum(dim=1) - 1  # (batch,)

                # Extract last token for each sequence in batch
                batch_acts = []
                for j, idx in enumerate(last_token_indices):
                    act = hidden[j, idx, :]  # (d,)
                    batch_acts.append(act)
                batch_acts = torch.stack(batch_acts)  # (batch, d)

            elif position == 'mean':
                # Mean pooling with attention mask
                attention_mask = inputs['attention_mask'].unsqueeze(-1)  # (batch, seq, 1)
                masked_hidden = hidden * attention_mask  # Zero out padding
                sum_hidden = masked_hidden.sum(dim=1)  # (batch, d)
                sum_mask = attention_mask.sum(dim=1)  # (batch, 1)
                batch_acts = sum_hidden / sum_mask  # (batch, d)

            activations.append(batch_acts.cpu())

        return torch.cat(activations, dim=0)  # (num_texts, d)

    def extract_persona_vector(
        self,
        positive_texts: List[str],
        negative_texts: List[str],
        position: Literal['last', 'mean'] = 'last'
    ) -> torch.Tensor:
        """
        Extract persona vector via contrastive mean difference.

        This is the standard method from steering vector papers:
        v = mean(activations on positive examples) - mean(activations on negative examples)

        Args:
            positive_texts: Texts exhibiting the target persona
            negative_texts: Texts exhibiting the opposite/neutral persona
            position: Extraction position (default: 'last')

        Returns:
            Normalized persona vector of shape (hidden_dim,)
        """
        print(f"Extracting persona vector from {len(positive_texts)} positive and {len(negative_texts)} negative examples...")

        # Extract activations for both sets
        pos_acts = self.extract_batch(positive_texts, position=position)  # (n_pos, d)
        neg_acts = self.extract_batch(negative_texts, position=position)  # (n_neg, d)

        # Contrastive mean
        pos_mean = pos_acts.mean(dim=0)  # (d,)
        neg_mean = neg_acts.mean(dim=0)  # (d,)

        vector = pos_mean - neg_mean  # (d,)

        # Normalize to unit norm
        vector = vector / vector.norm()

        print(f"Persona vector extracted: shape={vector.shape}, norm={vector.norm():.4f}")

        return vector

    def get_hidden_dim(self) -> int:
        """Get the hidden dimension of this model at the specified layer."""
        # Run a dummy forward pass to get dimension
        dummy_text = "Hello world"
        activation = self.extract(dummy_text)
        return activation.shape[0]


def sanity_check_extraction():
    """
    Quick sanity check to verify extraction works correctly.
    Run this before starting experiments.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("Running sanity check on extraction protocol...")

    # Load a small model for testing
    model_name = "gpt2"  # Small model for quick test
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Create extractor
    extractor = ActivationExtractor(model, tokenizer, layer_idx=6, device="cpu")

    # Test 1: Single extraction
    text = "The quick brown fox"
    act = extractor.extract(text, position='last')
    print(f"✓ Single extraction: shape={act.shape}")

    # Test 2: Batch extraction
    texts = ["Hello world", "How are you?", "This is a test"]
    acts = extractor.extract_batch(texts, position='last')
    print(f"✓ Batch extraction: shape={acts.shape}")
    assert acts.shape[0] == len(texts), "Batch size mismatch!"

    # Test 3: Persona vector extraction
    pos_texts = ["I am very happy!", "This is wonderful!", "I love this!"]
    neg_texts = ["I am sad.", "This is terrible.", "I hate this."]
    persona_vec = extractor.extract_persona_vector(pos_texts, neg_texts)
    print(f"✓ Persona vector: shape={persona_vec.shape}, norm={persona_vec.norm():.4f}")
    assert abs(persona_vec.norm().item() - 1.0) < 1e-5, "Vector not normalized!"

    # Test 4: Different positions give different results
    act_last = extractor.extract(text, position='last')
    act_mean = extractor.extract(text, position='mean')
    assert not torch.allclose(act_last, act_mean), "Last and mean should differ!"
    print(f"✓ Position parameter works correctly")

    print("\n✅ All sanity checks passed!")


if __name__ == "__main__":
    sanity_check_extraction()
