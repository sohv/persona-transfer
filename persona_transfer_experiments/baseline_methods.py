"""
Baseline Methods - Various approaches for transferring persona vectors.

This module implements multiple baseline methods to fairly compare against
the learned ridge mapping. Each method transforms a source persona vector
to the target model's dimension.

Baselines:
1. Zero-padding: Append zeros to match target dimension
2. Interpolation: Linearly interpolate to target dimension
3. Random projection: Random orthogonal matrix
4. Ridge mapping: Our learned method (for comparison)
5. Oracle: Native vector from target model (upper bound)
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict


class BaselineTransfer:
    """
    Collection of baseline methods for dimension transfer.
    """

    def __init__(self, d_source: int, d_target: int):
        """
        Args:
            d_source: Source model hidden dimension
            d_target: Target model hidden dimension
        """
        self.d_source = d_source
        self.d_target = d_target

        # Pre-compute random projection matrix (same for all vectors)
        self._random_proj = self._create_random_projection()

    def _create_random_projection(self) -> torch.Tensor:
        """
        Create random orthogonal projection matrix.

        This serves as a control baseline: random projection should NOT work,
        demonstrating that learned alignment captures meaningful structure.
        """
        # Generate random Gaussian matrix
        W = torch.randn(self.d_source, self.d_target) / np.sqrt(self.d_source)

        # Orthogonalize via QR decomposition (makes it isometric)
        Q, R = torch.linalg.qr(W)

        return Q

    def zero_pad(self, v_source: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """
        Naive method 1: Zero-pad to target dimension.

        Simply appends zeros: [v_source, 0, 0, ..., 0]

        Args:
            v_source: Source vector (d_source,)
            normalize: Whether to normalize output

        Returns:
            Padded vector (d_target,)
        """
        assert v_source.shape[0] == self.d_source, f"Expected {self.d_source}-d vector"

        if self.d_target > self.d_source:
            # Pad with zeros
            v_target = F.pad(v_source, (0, self.d_target - self.d_source))
        elif self.d_target < self.d_source:
            # Truncate
            v_target = v_source[:self.d_target]
        else:
            # Same dimension
            v_target = v_source.clone()

        if normalize:
            v_target = v_target / (v_target.norm() + 1e-8)

        return v_target

    def interpolate(self, v_source: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """
        Naive method 2: Linear interpolation to target dimension.

        Uses 1D interpolation to resample the vector.

        Args:
            v_source: Source vector (d_source,)
            normalize: Whether to normalize output

        Returns:
            Interpolated vector (d_target,)
        """
        assert v_source.shape[0] == self.d_source, f"Expected {self.d_source}-d vector"

        if self.d_source == self.d_target:
            v_target = v_source.clone()
        else:
            # Reshape for interpolate: (1, 1, d_source)
            v_reshaped = v_source.unsqueeze(0).unsqueeze(0)

            # Interpolate to target size
            v_interpolated = F.interpolate(
                v_reshaped,
                size=self.d_target,
                mode='linear',
                align_corners=True
            )

            # Reshape back: (d_target,)
            v_target = v_interpolated.squeeze(0).squeeze(0)

        if normalize:
            v_target = v_target / (v_target.norm() + 1e-8)

        return v_target

    def random_projection(self, v_source: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """
        Baseline method 3: Random orthogonal projection.

        Projects source vector through random orthogonal matrix.
        This should NOT work well (control baseline).

        Args:
            v_source: Source vector (d_source,)
            normalize: Whether to normalize output

        Returns:
            Randomly projected vector (d_target,)
        """
        assert v_source.shape[0] == self.d_source, f"Expected {self.d_source}-d vector"

        v_target = v_source @ self._random_proj  # (d_target,)

        if normalize:
            v_target = v_target / (v_target.norm() + 1e-8)

        return v_target

    def apply_all_baselines(
        self,
        v_source: torch.Tensor,
        v_oracle: torch.Tensor = None,
        ridge_mapping=None,
        normalize: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Apply all baseline methods and return as dictionary.

        Args:
            v_source: Source persona vector (d_source,)
            v_oracle: Optional native target vector (d_target,) for upper bound
            ridge_mapping: Optional RidgeMapping object for learned method
            normalize: Whether to normalize outputs

        Returns:
            Dictionary mapping method name to transferred vector
        """
        results = {
            'zero_pad': self.zero_pad(v_source, normalize=normalize),
            'interpolate': self.interpolate(v_source, normalize=normalize),
            'random_proj': self.random_projection(v_source, normalize=normalize),
        }

        # Add ridge mapping if provided
        if ridge_mapping is not None:
            results['ridge_mapped'] = ridge_mapping.transform(v_source, normalize=normalize)

        # Add oracle if provided
        if v_oracle is not None:
            if normalize:
                v_oracle_norm = v_oracle / (v_oracle.norm() + 1e-8)
            else:
                v_oracle_norm = v_oracle
            results['oracle_native'] = v_oracle_norm

        return results

    def compare_to_oracle(
        self,
        v_source: torch.Tensor,
        v_oracle: torch.Tensor,
        ridge_mapping=None
    ) -> Dict[str, float]:
        """
        Compare all methods to oracle (native target vector).

        Computes cosine similarity between each transferred vector and the oracle.
        Higher similarity = better transfer.

        Args:
            v_source: Source persona vector
            v_oracle: Native target persona vector (ground truth)
            ridge_mapping: Optional RidgeMapping object

        Returns:
            Dictionary of cosine similarities
        """
        # Get all transferred vectors
        transferred = self.apply_all_baselines(
            v_source,
            v_oracle=v_oracle,
            ridge_mapping=ridge_mapping,
            normalize=True
        )

        # Compute cosine similarity to oracle
        similarities = {}
        for method, v_transferred in transferred.items():
            if method == 'oracle_native':
                continue  # Skip self-comparison

            # Cosine similarity (vectors are normalized)
            sim = (v_transferred @ v_oracle).item()
            similarities[method] = sim

        # Oracle self-similarity should be 1.0
        similarities['oracle_native'] = 1.0

        return similarities


def analyze_baseline_properties(d_source: int = 3548, d_target: int = 4096):
    """
    Analyze mathematical properties of baseline methods.
    """
    print("=== Analyzing Baseline Transfer Methods ===\n")

    baseline = BaselineTransfer(d_source, d_target)

    # Create a unit random vector
    v_source = torch.randn(d_source)
    v_source = v_source / v_source.norm()

    print(f"Source vector: dim={d_source}, norm={v_source.norm():.4f}\n")

    # Apply all methods
    results = baseline.apply_all_baselines(v_source, normalize=True)

    print("Results:")
    for method, v_transferred in results.items():
        norm = v_transferred.norm().item()
        # Check how much "information" is preserved (via inner product)
        # For zero_pad: only first d_source dimensions matter
        if method == 'zero_pad' and d_target > d_source:
            overlap = (v_source @ v_transferred[:d_source]).item()
        else:
            overlap = "N/A"

        print(f"  {method:15s}: dim={v_transferred.shape[0]:4d}, norm={norm:.4f}, overlap={overlap}")

    # Test random projection consistency
    v_proj1 = baseline.random_projection(v_source)
    v_proj2 = baseline.random_projection(v_source)
    consistency = (v_proj1 @ v_proj2).item()
    print(f"\nRandom projection consistency (should be 1.0): {consistency:.4f}")

    # Simulate oracle comparison
    print("\n=== Simulated Oracle Comparison ===")
    print("(Assuming oracle is a random unit vector in target space)")

    v_oracle = torch.randn(d_target)
    v_oracle = v_oracle / v_oracle.norm()

    similarities = baseline.compare_to_oracle(v_source, v_oracle)

    print("\nCosine similarity to oracle:")
    for method, sim in sorted(similarities.items(), key=lambda x: x[1], reverse=True):
        print(f"  {method:15s}: {sim:.4f}")

    print("\nExpectation:")
    print("  - Oracle should be 1.0 (self-similarity)")
    print("  - Random projection should be ~0.0 (orthogonal)")
    print("  - Zero-pad and interpolate: unpredictable (depends on alignment)")
    print("  - Ridge mapped: should be highest among transfer methods")


class SteeringConfig:
    """
    Configuration for steering during generation.
    """

    def __init__(
        self,
        layer_idx: int,
        alpha: float = 1.0,
        normalize_vector: bool = True,
        add_at_every_token: bool = True
    ):
        """
        Args:
            layer_idx: Which layer to add steering vector
            alpha: Scaling coefficient (how strong the steering)
            normalize_vector: Whether to normalize steering vector to unit norm
            add_at_every_token: Whether to add at every generation step (vs. just first)
        """
        self.layer_idx = layer_idx
        self.alpha = alpha
        self.normalize_vector = normalize_vector
        self.add_at_every_token = add_at_every_token

    def __repr__(self):
        return (f"SteeringConfig(layer={self.layer_idx}, alpha={self.alpha}, "
                f"normalize={self.normalize_vector}, every_token={self.add_at_every_token})")


if __name__ == "__main__":
    # Test baseline methods
    analyze_baseline_properties(d_source=3548, d_target=4096)

    print("\n✅ Baseline methods implemented successfully!")
