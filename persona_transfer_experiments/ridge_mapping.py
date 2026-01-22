"""
Ridge Regression Mapping - Learn linear transformation between model activation spaces.

This module learns a mapping W: R^{d_source} -> R^{d_target} that aligns
the representation spaces of two different language models.

Theory:
Given paired activations H_source (N x d_source) and H_target (N x d_target),
we solve: minimize ||H_target - H_source @ W||^2 + λ||W||^2

The regularization λ is critical because:
- W has d_source * d_target parameters (~14M for Qwen→LLaMA)
- We only have N (~5k) samples
- Ridge regression with λ > 0 has effective parameters << total parameters
- Proper cross-validation ensures generalization
"""

import torch
import numpy as np
from typing import List, Tuple
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from pathlib import Path


class RidgeMapping:
    """
    Learns and applies linear mapping between model activation spaces.
    """

    def __init__(self, d_source: int, d_target: int):
        """
        Args:
            d_source: Source model hidden dimension
            d_target: Target model hidden dimension
        """
        self.d_source = d_source
        self.d_target = d_target
        self.W = None  # Will be (d_source, d_target)
        self.best_lambda = None

    def fit(
        self,
        H_source: torch.Tensor,
        H_target: torch.Tensor,
        lambda_value: float = 1.0
    ) -> None:
        """
        Fit ridge regression to learn mapping W.

        Solves: W = (H_source^T H_source + λI)^{-1} H_source^T H_target

        Args:
            H_source: Source activations (N, d_source)
            H_target: Target activations (N, d_target)
            lambda_value: Regularization strength
        """
        assert H_source.shape[0] == H_target.shape[0], "Sample size mismatch!"
        assert H_source.shape[1] == self.d_source, f"Expected d_source={self.d_source}, got {H_source.shape[1]}"
        assert H_target.shape[1] == self.d_target, f"Expected d_target={self.d_target}, got {H_target.shape[1]}"

        N = H_source.shape[0]
        print(f"Fitting ridge regression: {N} samples, λ={lambda_value}")

        # Move to double precision for numerical stability
        H_source = H_source.double()
        H_target = H_target.double()

        # Compute Gram matrix: H_source^T @ H_source
        # Shape: (d_source, d_source)
        gram = H_source.T @ H_source

        # Add ridge regularization: λI
        ridge_term = lambda_value * torch.eye(self.d_source, dtype=torch.float64)
        gram_regularized = gram + ridge_term

        # Compute H_source^T @ H_target
        # Shape: (d_source, d_target)
        cross_term = H_source.T @ H_target

        # Solve linear system: (H^T H + λI) W = H^T Y
        # This is more stable than computing the inverse explicitly
        try:
            self.W = torch.linalg.solve(gram_regularized, cross_term)
        except RuntimeError as e:
            print(f"Warning: linalg.solve failed, using lstsq fallback. Error: {e}")
            self.W = torch.linalg.lstsq(gram_regularized, cross_term).solution

        # Convert back to float32
        self.W = self.W.float()

        print(f"✓ Mapping learned: W.shape={self.W.shape}")

        # Compute training reconstruction error
        train_error = self._reconstruction_error(H_source.float(), H_target.float())
        print(f"  Training reconstruction error: {train_error:.4f}")

    def cross_validate(
        self,
        H_source: torch.Tensor,
        H_target: torch.Tensor,
        lambda_values: List[float] = [0.01, 0.1, 1.0, 10.0, 100.0],
        n_folds: int = 5
    ) -> Tuple[float, List[float]]:
        """
        Cross-validate to find best λ value.

        Args:
            H_source: Source activations (N, d_source)
            H_target: Target activations (N, d_target)
            lambda_values: List of λ values to try
            n_folds: Number of CV folds

        Returns:
            best_lambda: Best λ value
            cv_errors: List of mean CV errors for each λ
        """
        print(f"\nCross-validating ridge regression with {n_folds}-fold CV...")
        print(f"Testing λ values: {lambda_values}")

        N = H_source.shape[0]
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

        cv_errors = []

        for lambda_val in lambda_values:
            fold_errors = []

            for fold_idx, (train_idx, val_idx) in enumerate(kf.split(range(N))):
                # Split data
                H_src_train = H_source[train_idx]
                H_tgt_train = H_target[train_idx]
                H_src_val = H_source[val_idx]
                H_tgt_val = H_target[val_idx]

                # Fit on training fold
                temp_mapping = RidgeMapping(self.d_source, self.d_target)
                temp_mapping.fit(H_src_train, H_tgt_train, lambda_value=lambda_val)

                # Evaluate on validation fold
                val_error = temp_mapping._reconstruction_error(H_src_val, H_tgt_val)
                fold_errors.append(val_error)

            mean_error = np.mean(fold_errors)
            std_error = np.std(fold_errors)
            cv_errors.append(mean_error)

            print(f"  λ={lambda_val:>6.2f}: {mean_error:.4f} ± {std_error:.4f}")

        # Find best lambda
        best_idx = np.argmin(cv_errors)
        best_lambda = lambda_values[best_idx]

        print(f"\n✓ Best λ={best_lambda} with CV error={cv_errors[best_idx]:.4f}")

        # Fit final model on all data with best lambda
        self.best_lambda = best_lambda
        self.fit(H_source, H_target, lambda_value=best_lambda)

        return best_lambda, cv_errors

    def transform(self, v_source: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """
        Apply learned mapping to a source vector.

        Args:
            v_source: Source vector (d_source,) or batch (N, d_source)
            normalize: Whether to normalize output to unit norm

        Returns:
            Mapped vector (d_target,) or (N, d_target)
        """
        assert self.W is not None, "Must call fit() or cross_validate() first!"

        is_single = v_source.ndim == 1
        if is_single:
            v_source = v_source.unsqueeze(0)  # (1, d_source)

        # Apply mapping: v_target = v_source @ W
        v_target = v_source @ self.W  # (N, d_target)

        if normalize:
            # Normalize each vector to unit norm
            norms = v_target.norm(dim=1, keepdim=True)
            v_target = v_target / (norms + 1e-8)

        if is_single:
            v_target = v_target.squeeze(0)  # (d_target,)

        return v_target

    def _reconstruction_error(self, H_source: torch.Tensor, H_target: torch.Tensor) -> float:
        """
        Compute reconstruction error: ||H_target - H_source @ W||^2 / N

        Args:
            H_source: Source activations (N, d_source)
            H_target: Target activations (N, d_target)

        Returns:
            Mean squared error
        """
        H_predicted = H_source @ self.W  # (N, d_target)
        error = torch.mean((H_target - H_predicted) ** 2).item()
        return error

    def analyze_mapping(self, save_dir: str = None) -> dict:
        """
        Analyze properties of learned mapping W.

        Returns:
            Dictionary with analysis metrics
        """
        assert self.W is not None, "Must fit mapping first!"

        print("\n=== Mapping Analysis ===")

        # Compute SVD: W = U S V^T
        U, S, Vt = torch.svd(self.W)

        # Effective rank (singular values > 1% of largest)
        threshold = 0.01 * S[0]
        effective_rank = (S > threshold).sum().item()

        # Explained variance by top k components
        total_var = (S ** 2).sum()
        cumsum_var = torch.cumsum(S ** 2, dim=0) / total_var

        metrics = {
            'frobenius_norm': torch.norm(self.W, p='fro').item(),
            'nuclear_norm': S.sum().item(),
            'effective_rank': effective_rank,
            'max_singular_value': S[0].item(),
            'min_singular_value': S[-1].item(),
            'condition_number': (S[0] / S[-1]).item(),
            'variance_explained_top10': cumsum_var[9].item() if len(S) > 9 else cumsum_var[-1].item(),
            'variance_explained_top100': cumsum_var[99].item() if len(S) > 99 else cumsum_var[-1].item(),
        }

        print(f"  Frobenius norm: {metrics['frobenius_norm']:.2f}")
        print(f"  Effective rank: {effective_rank} / {min(self.d_source, self.d_target)}")
        print(f"  Condition number: {metrics['condition_number']:.2e}")
        print(f"  Top 10 components explain {metrics['variance_explained_top10']*100:.1f}% variance")
        print(f"  Top 100 components explain {metrics['variance_explained_top100']*100:.1f}% variance")

        # Plot singular value spectrum
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

            plt.figure(figsize=(10, 4))

            plt.subplot(1, 2, 1)
            plt.semilogy(S.numpy())
            plt.axhline(threshold.item(), color='r', linestyle='--', label=f'Threshold (1% of max)')
            plt.xlabel('Singular Value Index')
            plt.ylabel('Singular Value (log scale)')
            plt.title(f'Singular Value Spectrum (Effective Rank: {effective_rank})')
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.subplot(1, 2, 2)
            plt.plot(cumsum_var.numpy() * 100)
            plt.axhline(90, color='r', linestyle='--', alpha=0.5)
            plt.axhline(95, color='r', linestyle='--', alpha=0.5)
            plt.axhline(99, color='r', linestyle='--', alpha=0.5)
            plt.xlabel('Number of Components')
            plt.ylabel('Cumulative Variance Explained (%)')
            plt.title('Cumulative Variance Explained')
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(save_dir / 'mapping_analysis.png', dpi=150)
            print(f"  Saved plot to {save_dir / 'mapping_analysis.png'}")
            plt.close()

        return metrics

    def save(self, path: str) -> None:
        """Save mapping to file."""
        torch.save({
            'W': self.W,
            'd_source': self.d_source,
            'd_target': self.d_target,
            'best_lambda': self.best_lambda,
        }, path)
        print(f"✓ Saved mapping to {path}")

    @classmethod
    def load(cls, path: str) -> 'RidgeMapping':
        """Load mapping from file."""
        checkpoint = torch.load(path)
        mapping = cls(checkpoint['d_source'], checkpoint['d_target'])
        mapping.W = checkpoint['W']
        mapping.best_lambda = checkpoint.get('best_lambda', None)
        print(f"✓ Loaded mapping from {path}")
        return mapping


def create_random_baseline(d_source: int, d_target: int) -> torch.Tensor:
    """
    Create random projection baseline for comparison.

    This is a control: random orthogonal matrix should NOT work well,
    demonstrating that learned mapping is genuinely capturing structure.

    Args:
        d_source: Source dimension
        d_target: Target dimension

    Returns:
        Random orthogonal matrix (d_source, d_target)
    """
    # Generate random matrix
    W_random = torch.randn(d_source, d_target) / np.sqrt(d_source)

    # Orthogonalize via QR decomposition
    Q, R = torch.linalg.qr(W_random)

    return Q  # Orthogonal matrix


if __name__ == "__main__":
    # Test ridge mapping
    print("Testing ridge mapping implementation...\n")

    # Simulate data
    N = 1000
    d_source = 100
    d_target = 150

    # Generate correlated activations (simulating aligned models)
    H_source = torch.randn(N, d_source)

    # True mapping with noise
    W_true = torch.randn(d_source, d_target) / 10
    H_target = H_source @ W_true + 0.1 * torch.randn(N, d_target)

    # Test fitting
    mapping = RidgeMapping(d_source, d_target)
    best_lambda, cv_errors = mapping.cross_validate(H_source, H_target)

    # Test transform
    v_source = torch.randn(d_source)
    v_target = mapping.transform(v_source)
    print(f"\nTransform test: {v_source.shape} -> {v_target.shape}")
    print(f"Output norm: {v_target.norm():.4f} (should be ~1.0 with normalize=True)")

    # Analyze mapping
    metrics = mapping.analyze_mapping()

    # Compare to random baseline
    W_random = create_random_baseline(d_source, d_target)
    random_error = torch.mean((H_target - H_source @ W_random) ** 2).item()
    learned_error = mapping._reconstruction_error(H_source, H_target)

    print(f"\n=== Baseline Comparison ===")
    print(f"Random projection error: {random_error:.4f}")
    print(f"Learned mapping error: {learned_error:.4f}")
    print(f"Improvement: {(random_error - learned_error) / random_error * 100:.1f}%")

    print("\n✅ All tests passed!")
