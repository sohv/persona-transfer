"""
Dimension mapping for cross-model persona vector transfer.

Handles dimension mismatch when transferring persona vectors between models
with different hidden sizes using Procrustes alignment.
"""

import json
import logging
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Tuple, Optional
from scipy.linalg import orthogonal_procrustes

logger = logging.getLogger(__name__)

# Cache directory for mappings
MAPPINGS_DIR = Path(__file__).parent / "data" / "mappings"
MAPPINGS_DIR.mkdir(parents=True, exist_ok=True)


class DimensionMapper:
    """
    Maps persona vectors from source model dimension to target model dimension.

    Uses Procrustes alignment to find optimal orthogonal transformation when
    dimensions match, or learned linear projection when dimensions differ.
    """

    def __init__(self):
        self.mappings_cache = {}

    def get_mapping_key(self, source_model: str, target_model: str, source_dim: int, target_dim: int) -> str:
        """Generate cache key for model pair mapping."""
        return f"{source_model}_{source_dim}_{target_model}_{target_dim}"

    def load_mapping(self, source_model: str, target_model: str, source_dim: int, target_dim: int) -> Optional[Dict]:
        """Load cached mapping from disk."""
        key = self.get_mapping_key(source_model, target_model, source_dim, target_dim)

        # Check memory cache first
        if key in self.mappings_cache:
            return self.mappings_cache[key]

        # Check disk cache
        mapping_file = MAPPINGS_DIR / f"{key}.json"
        if mapping_file.exists():
            try:
                with open(mapping_file, 'r') as f:
                    data = json.load(f)

                # Convert lists back to numpy arrays
                if 'W' in data:
                    data['W'] = np.array(data['W'])
                if 'bias' in data:
                    data['bias'] = np.array(data['bias'])

                self.mappings_cache[key] = data
                logger.info(f"Loaded cached mapping: {key}")
                return data
            except Exception as e:
                logger.warning(f"Failed to load cached mapping {key}: {e}")

        return None

    def save_mapping(self, source_model: str, target_model: str, source_dim: int, target_dim: int, mapping_data: Dict):
        """Save mapping to disk cache."""
        key = self.get_mapping_key(source_model, target_model, source_dim, target_dim)

        # Store in memory cache
        self.mappings_cache[key] = mapping_data

        # Save to disk
        mapping_file = MAPPINGS_DIR / f"{key}.json"
        try:
            # Convert numpy arrays to lists for JSON serialization
            save_data = {}
            for k, v in mapping_data.items():
                if isinstance(v, np.ndarray):
                    save_data[k] = v.tolist()
                else:
                    save_data[k] = v

            with open(mapping_file, 'w') as f:
                json.dump(save_data, f)

            logger.info(f"Saved mapping to {mapping_file}")
        except Exception as e:
            logger.error(f"Failed to save mapping {key}: {e}")

    def compute_procrustes_mapping(
        self,
        source_activations: np.ndarray,
        target_activations: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Compute Procrustes alignment between source and target activations.

        Args:
            source_activations: (n_samples, source_dim)
            target_activations: (n_samples, target_dim)

        Returns:
            W: Transformation matrix (target_dim, source_dim)
            scale: Optimal scale factor
        """
        # Center the data
        source_mean = np.mean(source_activations, axis=0)
        target_mean = np.mean(target_activations, axis=0)

        source_centered = source_activations - source_mean
        target_centered = target_activations - target_mean

        # Handle dimension mismatch
        source_dim = source_centered.shape[1]
        target_dim = target_centered.shape[1]

        if source_dim == target_dim:
            # Same dimensions: Use standard orthogonal Procrustes
            W, scale = orthogonal_procrustes(source_centered, target_centered)
            return W.T, scale  # Transpose to get (target_dim, source_dim)

        elif source_dim < target_dim:
            # Source smaller: Learn projection to higher dimension
            # Use least squares: W = argmin ||W @ X_source - X_target||²
            # Solution: W = X_target.T @ X_source @ (X_source.T @ X_source)^-1

            # Add regularization to avoid overfitting
            reg = 1e-6 * np.eye(source_dim)
            W = target_centered.T @ source_centered @ np.linalg.inv(
                source_centered.T @ source_centered + reg
            )

            # Compute scale factor
            projected = source_centered @ W.T
            scale = np.linalg.norm(target_centered) / np.linalg.norm(projected)

            return W, scale

        else:  # source_dim > target_dim
            # Source larger: Project down to lower dimension
            # Use least squares: W = argmin ||W @ X_source - X_target||²

            reg = 1e-6 * np.eye(source_dim)
            W = target_centered.T @ source_centered @ np.linalg.inv(
                source_centered.T @ source_centered + reg
            )

            projected = source_centered @ W.T
            scale = np.linalg.norm(target_centered) / np.linalg.norm(projected)

            return W, scale

    def collect_aligned_vectors_from_files(
        self,
        source_model_id: str,
        target_model_id: str,
        num_samples: int = 50
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Collect aligned vectors from existing persona vector files.

        Uses the persona vectors for the same traits across different models
        as aligned samples for Procrustes alignment.

        Args:
            source_model_id: Source model identifier
            target_model_id: Target model identifier
            num_samples: Number of aligned samples to collect

        Returns:
            source_vecs: (num_samples, source_dim)
            target_vecs: (num_samples, target_dim)
        """
        from persona_vectors import VECTORS_DIR

        logger.info(f"Collecting aligned vectors from files: {source_model_id} → {target_model_id}")

        source_vectors = []
        target_vectors = []

        # Find all traits that have vectors for both models
        source_files = list(VECTORS_DIR.glob(f"{source_model_id}_*.json"))
        traits = [f.stem.replace(f"{source_model_id}_", "") for f in source_files]

        logger.info(f"Found {len(traits)} traits with source vectors: {traits}")

        for trait in traits:
            # Check if target has this trait too
            target_file = VECTORS_DIR / f"{target_model_id}_{trait}.json"
            source_file = VECTORS_DIR / f"{source_model_id}_{trait}.json"

            if not target_file.exists():
                logger.warning(f"Target model missing trait: {trait}")
                continue

            try:
                # Load both vector files
                with open(source_file) as f:
                    source_data = json.load(f)
                with open(target_file) as f:
                    target_data = json.load(f)

                source_vecs = source_data.get('vectors', {})
                target_vecs = target_data.get('vectors', {})

                if not source_vecs or not target_vecs:
                    continue

                # Use corresponding layers from both models
                # Match by layer index (not name, since they might differ)
                source_layers = sorted(source_vecs.keys(), key=lambda x: int(x.split('_')[1]) if '_' in x else 0)
                target_layers = sorted(target_vecs.keys(), key=lambda x: int(x.split('_')[1]) if '_' in x else 0)

                # Sample evenly across layers
                num_layers = min(len(source_layers), len(target_layers))
                sample_indices = np.linspace(0, num_layers-1, min(num_layers, 10), dtype=int)

                for idx in sample_indices:
                    if idx < len(source_layers) and idx < len(target_layers):
                        src_vec = np.array(source_vecs[source_layers[idx]])
                        tgt_vec = np.array(target_vecs[target_layers[idx]])

                        # Ensure 1D
                        if len(src_vec.shape) > 1:
                            src_vec = src_vec.reshape(-1)
                        if len(tgt_vec.shape) > 1:
                            tgt_vec = tgt_vec.reshape(-1)

                        source_vectors.append(src_vec)
                        target_vectors.append(tgt_vec)

                logger.info(f"  Trait '{trait}': collected {len(sample_indices)} layer pairs")

            except Exception as e:
                logger.warning(f"Failed to load vectors for trait {trait}: {e}")
                continue

        if len(source_vectors) == 0 or len(target_vectors) == 0:
            raise ValueError(f"Failed to collect any aligned vectors. "
                           f"Ensure both {source_model_id} and {target_model_id} have persona vectors.")

        source_vecs = np.stack(source_vectors)
        target_vecs = np.stack(target_vectors)

        # Subsample if we have more than requested
        if len(source_vecs) > num_samples:
            indices = np.random.choice(len(source_vecs), num_samples, replace=False)
            source_vecs = source_vecs[indices]
            target_vecs = target_vecs[indices]

        logger.info(f"Collected {len(source_vecs)} aligned vector pairs")
        logger.info(f"Source shape: {source_vecs.shape}, Target shape: {target_vecs.shape}")

        return source_vecs, target_vecs

    def create_mapping(
        self,
        source_model_id: str,
        target_model_id: str,
        source_dim: int,
        target_dim: int,
        num_samples: int = 50
    ) -> Dict:
        """
        Create a new mapping between source and target models.

        Args:
            source_model_id: Source model identifier
            target_model_id: Target model identifier
            source_dim: Source hidden dimension
            target_dim: Target hidden dimension
            num_samples: Number of samples for alignment

        Returns:
            Mapping dictionary with 'W', 'bias', 'scale', and metadata
        """
        logger.info(f"Creating mapping: {source_model_id}({source_dim}) → {target_model_id}({target_dim})")

        # Collect aligned vectors from existing persona vector files
        source_vecs, target_vecs = self.collect_aligned_vectors_from_files(
            source_model_id, target_model_id, num_samples
        )

        # Compute Procrustes mapping
        W, scale = self.compute_procrustes_mapping(source_vecs, target_vecs)

        logger.info(f"Computed mapping matrix: {W.shape}, scale: {scale:.4f}")

        # Compute mapping quality metrics
        # Project source to target space
        projected = source_vecs @ W.T

        # Cosine similarity between projected and target
        similarities = []
        for i in range(len(projected)):
            proj_norm = np.linalg.norm(projected[i])
            tgt_norm = np.linalg.norm(target_vecs[i])
            if proj_norm > 0 and tgt_norm > 0:
                cos_sim = np.dot(projected[i], target_vecs[i]) / (proj_norm * tgt_norm)
                similarities.append(cos_sim)
        avg_similarity = np.mean(similarities) if similarities else 0.0

        # MSE
        mse = np.mean((projected - target_vecs) ** 2)

        logger.info(f"Mapping quality: cosine_sim={avg_similarity:.4f}, mse={mse:.6f}")

        mapping_data = {
            'W': W,
            'bias': np.zeros(target_dim),  # Could compute optimal bias if needed
            'scale': float(scale),
            'source_model': source_model_id,
            'target_model': target_model_id,
            'source_dim': source_dim,
            'target_dim': target_dim,
            'num_samples': num_samples,
            'quality_metrics': {
                'cosine_similarity': float(avg_similarity),
                'mse': float(mse)
            }
        }

        # Save to cache
        self.save_mapping(source_model_id, target_model_id, source_dim, target_dim, mapping_data)

        return mapping_data

    def apply_mapping(
        self,
        vectors: Dict[str, np.ndarray],
        source_model: str,
        target_model: str,
        source_dim: int,
        target_dim: int
    ) -> Dict[str, np.ndarray]:
        """
        Apply dimension mapping to persona vectors.

        Args:
            vectors: Dict of layer_name -> vector (source_dim,)
            source_model: Source model ID
            target_model: Target model ID
            source_dim: Source dimension
            target_dim: Target dimension

        Returns:
            Dict of layer_name -> mapped_vector (target_dim,)
        """
        if source_dim == target_dim:
            logger.info("Dimensions match, no mapping needed")
            return vectors

        # Load mapping
        mapping = self.load_mapping(source_model, target_model, source_dim, target_dim)

        if mapping is None:
            raise ValueError(
                f"No mapping found for {source_model}({source_dim}) → {target_model}({target_dim}). "
                f"Create mapping first with: python create_mapping.py --source {source_model} --target {target_model}"
            )

        W = mapping['W']
        bias = mapping.get('bias', np.zeros(target_dim))

        logger.info(f"Applying mapping: {source_dim} → {target_dim}")

        # Apply mapping to each layer vector
        mapped_vectors = {}
        for layer_name, vector in vectors.items():
            # Ensure vector is 1D
            if len(vector.shape) > 1:
                vector = vector.reshape(-1)

            # Apply transformation: v_target = W @ v_source + bias
            mapped_vector = W @ vector + bias

            # Preserve norm (important for steering strength)
            original_norm = np.linalg.norm(vector)
            mapped_norm = np.linalg.norm(mapped_vector)
            if mapped_norm > 0:
                mapped_vector = mapped_vector * (original_norm / mapped_norm)

            mapped_vectors[layer_name] = mapped_vector

        logger.info(f"Mapped {len(mapped_vectors)} layer vectors")

        return mapped_vectors


# Global mapper instance
_mapper = DimensionMapper()


def get_dimension_mapper() -> DimensionMapper:
    """Get the global dimension mapper instance."""
    return _mapper
