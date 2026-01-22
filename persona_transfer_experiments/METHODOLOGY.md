# Persona Transfer Between Language Models: Complete Methodology

## Overview

This document provides comprehensive documentation for the persona transfer experiment, including theoretical foundations, implementation details, and usage instructions.

**Research Question:** Can we transfer persona vectors extracted from one language model (source) to another model (target) by learning a linear mapping between their activation spaces?

**Key Innovation:** Using ridge regression to learn a representation mapping that preserves semantic structure while adapting to dimensional differences.

---

## Table of Contents

1. [Theoretical Foundation](#theoretical-foundation)
2. [Architecture Overview](#architecture-overview)
3. [Core Components](#core-components)
4. [Experimental Pipeline](#experimental-pipeline)
5. [Mathematical Details](#mathematical-details)
6. [Implementation Guide](#implementation-guide)
7. [Expected Results](#expected-results)
8. [Troubleshooting](#troubleshooting)

---

## Theoretical Foundation

### The Problem

**Persona vectors** are directions in a language model's activation space that encode specific behavioral traits (e.g., "formal", "silly", "concise"). These vectors are extracted via contrastive learning on paired examples.

When we extract a persona vector from model A (e.g., Qwen-7B), we cannot directly apply it to model B (e.g., LLaMA-2-7B) because:

1. **Dimensional mismatch**: Qwen has hidden dim 3548, LLaMA has 4096
2. **Semantic misalignment**: Even at matching dimensions, the basis vectors represent different features
3. **Naive solutions fail**: Zero-padding or interpolation don't respect the semantic structure

### The Hypothesis

If two models learn similar representations (as suggested by emergent phenomena in LLMs), there should exist a **linear transformation** `W` that aligns their activation spaces:

```
h_target ≈ h_source @ W
```

By learning `W` from paired activations on neutral text, we can then map persona vectors:

```
v_target = v_source @ W
```

### Why Ridge Regression?

**Ridge regression** solves:

```
minimize ||H_target - H_source @ W||² + λ||W||²
```

Where:
- `H_source`: (N, d_source) matrix of source activations
- `H_target`: (N, d_target) matrix of target activations
- `W`: (d_source, d_target) transformation matrix
- `λ`: Regularization strength

**Key properties:**

1. **Closed-form solution**: No training loop needed
   ```
   W = (H_source^T H_source + λI)^{-1} H_source^T H_target
   ```

2. **Effective regularization**: With λ > 0, the effective parameters << total parameters
   - W has 14.5M parameters (3548 × 4096)
   - But with λ=1.0, effective rank ~ 1000-2000
   - Only need ~5k samples, not 14M

3. **Interpretable**: Can analyze via SVD to understand learned structure

---

## Architecture Overview

### File Structure

```
persona_transfer_experiments/
├── activation_extractor.py      # Consistent activation extraction protocol
├── ridge_mapping.py             # Ridge regression implementation
├── baseline_methods.py          # Naive transfer baselines
├── steered_generation.py        # Text generation with steering
├── evaluation.py                # Coherence and trait metrics
├── run_experiment.py            # Main experiment orchestrator
├── example_config.json          # Configuration template
└── METHODOLOGY.md              # This document
```

### Data Flow

```
┌─────────────────┐
│   WikiText-103  │ (neutral text for learning mapping)
└────────┬────────┘
         │
         ↓
┌─────────────────────────────────────────────────────┐
│   Extract Paired Activations                        │
│   • Run same text through Qwen and LLaMA            │
│   • Extract at layer 16, last token position        │
│   • H_qwen: (5k, 3548), H_llama: (5k, 4096)         │
└────────┬────────────────────────────────────────────┘
         │
         ↓
┌─────────────────────────────────────────────────────┐
│   Learn Ridge Mapping                                │
│   • Cross-validate λ ∈ {0.01, 0.1, 1, 10, 100}      │
│   • Fit W: (3548, 4096)                              │
│   • Analyze via SVD                                  │
└────────┬────────────────────────────────────────────┘
         │
         ↓
┌─────────────────────────────────────────────────────┐
│   Extract Persona Vectors                            │
│   • From Qwen: v_qwen = mean(pos) - mean(neg)        │
│   • From LLaMA (oracle): v_llama_native              │
│   • Apply baselines: zero-pad, interp, random        │
│   • Apply learned: v_llama_mapped = v_qwen @ W       │
└────────┬────────────────────────────────────────────┘
         │
         ↓
┌─────────────────────────────────────────────────────┐
│   Generate Text                                      │
│   • 5 conditions × 3 personas × 30 prompts × 3 seeds │
│   • Conditions: baseline, naive methods, mapped, oracle │
│   • Add steering: h'_layer = h_layer + α * v         │
└────────┬────────────────────────────────────────────┘
         │
         ↓
┌─────────────────────────────────────────────────────┐
│   Evaluate                                           │
│   • Coherence: perplexity, repetition collapse       │
│   • Trait strength: GPT-4 judge (optional)           │
│   • Aggregate by condition                           │
└─────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. ActivationExtractor (`activation_extractor.py`)

**Purpose:** Ensures consistent extraction protocol across all experiments.

**Critical design decisions:**

1. **Last token extraction:** Matches where steering is applied during generation
   ```python
   # During generation:
   h = model(prompt + generated_so_far)[:, -1, :]  # Last token
   h_steered = h + α * v  # Add steering vector
   ```

2. **Same layer everywhere:** Vector extraction and mapping must use same layer
   - If mapping learned at layer 16, extract personas at layer 16
   - Don't mix layer 16 map with layer 20 vectors

3. **Normalization:** All persona vectors normalized to unit L2 norm
   ```python
   v = (mean(positive) - mean(negative))
   v = v / ||v||
   ```

**Key methods:**

```python
extractor = ActivationExtractor(model, tokenizer, layer_idx=16, device="cuda")

# Single text extraction
activation = extractor.extract(text, position='last')  # (d,)

# Batch extraction (for training data)
activations = extractor.extract_batch(texts, position='last')  # (N, d)

# Persona vector extraction
v_persona = extractor.extract_persona_vector(
    positive_texts=["I'm so happy!", "This is wonderful!"],
    negative_texts=["I'm sad.", "This is terrible."]
)
```

**Handles different model architectures:**
- LLaMA/Mistral: `model.model.layers[i]`
- GPT-2: `model.transformer.h[i]`
- Qwen: Auto-detected

---

### 2. RidgeMapping (`ridge_mapping.py`)

**Purpose:** Learn linear transformation between activation spaces.

**Implementation highlights:**

1. **Numerical stability:** Uses double precision and `linalg.solve` (not matrix inversion)
   ```python
   gram = H_source.T @ H_source  # (d_s, d_s)
   ridge_term = λ * I  # (d_s, d_s)
   cross_term = H_source.T @ H_target  # (d_s, d_t)

   W = solve(gram + ridge_term, cross_term)  # Stable
   ```

2. **Cross-validation:** Selects best λ via 5-fold CV on reconstruction error
   ```python
   best_lambda, cv_errors = mapping.cross_validate(
       H_source, H_target,
       lambda_values=[0.01, 0.1, 1.0, 10.0, 100.0]
   )
   ```

3. **Analysis via SVD:** Understand learned structure
   ```python
   U, S, Vt = torch.svd(W)
   effective_rank = (S > 0.01 * S[0]).sum()  # Rank of meaningful components
   ```

**Why 5k samples is sufficient:**

With regularization, effective parameters ≈ effective rank, not total parameters:

| λ value | Effective Rank | Required Samples |
|---------|---------------|------------------|
| 0.01    | ~3000         | ~10k samples     |
| 1.0     | ~1000         | ~5k samples ✓    |
| 10.0    | ~300          | ~2k samples      |

---

### 3. BaselineTransfer (`baseline_methods.py`)

**Purpose:** Implement naive baselines for fair comparison.

**Methods:**

1. **Zero-padding:** `[v_source, 0, 0, ..., 0]`
   - Simple, preserves source information in first dims
   - May work partially if models align naturally

2. **Interpolation:** Linear resampling to target dimension
   - `F.interpolate(v_source, size=d_target)`
   - Smooth but arbitrary

3. **Random projection:** Multiply by random orthogonal matrix
   - Control baseline: should NOT work
   - Proves learned mapping captures structure

4. **Ridge mapping:** Our method
   - `v_target = v_source @ W`

5. **Oracle:** Native extraction from target
   - Upper bound on performance
   - Shows ceiling for transfer

**Usage:**

```python
baseline = BaselineTransfer(d_source=3548, d_target=4096)

# Apply all methods
transferred = baseline.apply_all_baselines(
    v_source=v_qwen,
    v_oracle=v_llama_native,
    ridge_mapping=mapping,
    normalize=True
)

# Compare to oracle
similarities = baseline.compare_to_oracle(v_source, v_oracle, mapping)
# Output: {'zero_pad': 0.23, 'random_proj': 0.01, 'ridge_mapped': 0.61, ...}
```

---

### 4. SteeredGenerator (`steered_generation.py`)

**Purpose:** Generate text with persona steering via activation patching.

**Mechanism:**

Uses PyTorch forward hooks to intercept and modify hidden states:

```python
# Hook function
def steering_hook(module, input, output):
    hidden_states = output[0]  # (batch, seq, d)
    steered = hidden_states + α * steering_vector  # Broadcast
    return (steered, output[1:])

# Register at specific layer
generator.register_steering(
    layer_idx=16,
    steering_vector=v_persona,
    alpha=2.0
)

# Generate (steering active during forward pass)
output = generator.generate(prompt, max_new_tokens=100)
```

**Key features:**

- **Multiple hooks:** Can steer at multiple layers simultaneously
- **Dynamic α:** Change steering strength without re-registering
- **Deactivate/activate:** Temporarily disable without removing hooks
- **Batch generation:** Process multiple prompts efficiently

---

### 5. Evaluator (`evaluation.py`)

**Purpose:** Measure quality of steered generation.

**Metrics:**

1. **Perplexity:** Model's confidence in its output
   - Computed as `exp(negative_log_likelihood)`
   - Lower = more fluent/expected
   - Uses target model itself as judge

2. **Repetition collapse:** Detect n-gram repetition
   - Compute unique n-gram ratio
   - Flag if >50% repetition
   - Common failure mode for naive transfer

3. **GPT-4 coherence judge:** Rate fluency 1-5
   - 1 = Gibberish, 5 = Perfect
   - Gold standard from steering literature
   - Optional (requires API key)

4. **GPT-4 trait judge:** Rate trait strength 1-5
   - 1 = Trait absent, 5 = Trait very strong
   - Validates steering worked

**Usage:**

```python
evaluator = Evaluator(model, tokenizer, use_gpt4=False)

result = evaluator.evaluate_single(
    prompt="Tell me about AI",
    output="AI is super cool and amazing! It's gonna change the world!",
    condition="ridge_mapped",
    persona="silly"
)

print(f"Perplexity: {result.perplexity:.2f}")
print(f"Collapsed: {result.is_collapsed}")
print(f"GPT-4 coherence: {result.gpt4_coherence_score}")
```

---

## Experimental Pipeline

### Phase 1: Setup (30 minutes)

1. **Install dependencies:**
   ```bash
   pip install torch transformers datasets openai scikit-learn matplotlib
   ```

2. **Download models:**
   ```python
   # Requires HuggingFace authentication for gated models
   from transformers import AutoModelForCausalLM

   qwen = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B")
   llama = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
   ```

3. **Verify extraction:**
   ```bash
   python activation_extractor.py  # Runs sanity checks
   ```

### Phase 2: Learn Mapping (2-3 hours)

1. **Collect training data:**
   - Load 5k texts from WikiText-103
   - Extract paired activations at layer 16, last token
   - Save to disk

2. **Cross-validate ridge regression:**
   - Test λ ∈ {0.01, 0.1, 1.0, 10.0, 100.0}
   - 5-fold CV on reconstruction error
   - Select best λ

3. **Analyze mapping:**
   - Compute SVD of W
   - Check effective rank
   - Plot singular value spectrum
   - Verify improvement over random baseline

**Expected output:**
- `training_data.pt`: Paired activations
- `ridge_mapping.pt`: Learned W matrix
- `cv_results.json`: Cross-validation errors
- `mapping_analysis.png`: SVD visualization

### Phase 3: Extract Personas (1 hour)

1. **Define persona datasets:**
   - 5-10 positive examples (exhibiting trait)
   - 5-10 negative examples (opposite/neutral)

2. **Extract from both models:**
   - Source (Qwen): `v_qwen = mean(pos) - mean(neg)`
   - Target (LLaMA): `v_llama_native = mean(pos) - mean(neg)`

3. **Apply all transfer methods:**
   - Zero-pad, interpolate, random projection
   - Ridge mapping: `v_llama_mapped = v_qwen @ W`

4. **Compute similarities to oracle:**
   - Best method should have highest cosine similarity to native vector

**Expected output:**
- `persona_vectors.pt`: All extracted vectors
- Console: Similarity scores for each method

### Phase 4: Generate (4-6 hours)

**Configuration:**
- 3 personas (silly, formal, concise)
- 5 conditions (baseline, zero_pad, interpolate, random_proj, ridge_mapped, oracle)
- 30 test prompts
- 5 α values (0.5, 1.0, 2.0, 3.0, 5.0)
- 3 random seeds

**Total:** 3 × 5 × 30 × 5 × 3 = 6,750 generations

**Parallelization tips:**
- Batch prompts when possible
- Use FP16 for faster generation
- Consider multiple GPUs

**Expected output:**
- `generation_results.json`: All generations with metadata

### Phase 5: Evaluate (2-3 hours)

1. **Compute automatic metrics:**
   - Perplexity (uses target model, ~30 sec per generation)
   - Repetition detection (fast, regex-based)
   - Output length statistics

2. **Optional: GPT-4 judging:**
   - Coherence scores (1-5)
   - Trait strength scores (1-5)
   - Expensive: ~$0.03 per generation × 6750 = $200
   - Consider sampling subset for cost

3. **Aggregate results:**
   - Group by condition
   - Compute means and stds
   - Create comparison tables

**Expected output:**
- `eval_results.json`: Per-generation metrics
- `aggregated_results.json`: Means by condition

---

## Mathematical Details

### Ridge Regression Derivation

**Problem:**
```
minimize_{W} ||H_target - H_source W||²_F + λ||W||²_F
```

**Solution:**

Taking the derivative with respect to W and setting to zero:

```
∂/∂W [||H_target - H_source W||² + λ||W||²] = 0

∂/∂W [tr((H_target - H_source W)^T (H_target - H_source W)) + λ tr(W^T W)] = 0

-2 H_source^T (H_target - H_source W) + 2λW = 0

H_source^T H_target = H_source^T H_source W + λW

H_source^T H_target = (H_source^T H_source + λI) W

W = (H_source^T H_source + λI)^{-1} H_source^T H_target
```

**Computational complexity:**
- Gram matrix: O(N d_s²) where N=5k, d_s=3548
- Matrix inversion: O(d_s³) ≈ 45 billion ops
- Cross term: O(N d_s d_t) where d_t=4096
- Total: ~1-2 minutes on GPU

### Effective Degrees of Freedom

The ridge estimator has effective DOF:

```
df(λ) = tr(H_source (H_source^T H_source + λI)^{-1} H_source^T)
       = Σ_i σ_i² / (σ_i² + λ)
```

Where σ_i are singular values of H_source.

For λ=1.0, this is typically ~1000-2000, much less than the 14M parameters.

### Why Last Token?

**During generation**, the model processes text autoregressively:

```
for t in range(max_tokens):
    h_t = model(x_1, ..., x_t)[:, -1, :]  # Last token embedding
    logits_t = lm_head(h_t + α * v_steer)  # Steering applied here
    x_{t+1} = sample(logits_t)
```

So our training distribution (last token of prompts) matches the application distribution (last token during generation).

**Alternative (mean pooling)** would mismatch:
- Train: mean across prompt tokens
- Apply: only last token during generation
- Mismatch could reduce effectiveness

---

## Implementation Guide

### Quick Start

1. **Create config file** (see `example_config.json`):
   ```json
   {
     "source_model": "Qwen/Qwen-7B",
     "target_model": "meta-llama/Llama-2-7b-hf",
     "layer_idx": 16,
     "output_dir": "./results/qwen_to_llama",
     "n_training_samples": 5000,
     "n_test_prompts": 30
   }
   ```

2. **Run experiment:**
   ```bash
   python run_experiment.py --config example_config.json
   ```

3. **Monitor progress:**
   ```bash
   tail -f results/qwen_to_llama/experiment.log
   ```

### Advanced Usage

**Test single component:**

```python
# Test activation extraction
from activation_extractor import ActivationExtractor
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
extractor = ActivationExtractor(model, tokenizer, layer_idx=6, device="cpu")

activation = extractor.extract("Hello world")
print(f"Shape: {activation.shape}, Norm: {activation.norm():.4f}")
```

**Test mapping:**

```python
# Simulate learning a mapping
from ridge_mapping import RidgeMapping
import torch

H_source = torch.randn(1000, 100)
H_target = torch.randn(1000, 150)

mapping = RidgeMapping(100, 150)
best_lambda, _ = mapping.cross_validate(H_source, H_target)

v_source = torch.randn(100)
v_target = mapping.transform(v_source)
print(f"Mapped: {v_source.shape} -> {v_target.shape}")
```

**Test generation:**

```python
from steered_generation import SteeredGenerator
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
generator = SteeredGenerator(model, tokenizer, device="cpu")

# Create random steering vector
v_steer = torch.randn(model.config.n_embd)
generator.register_steering(layer_idx=6, steering_vector=v_steer, alpha=1.0)

output = generator.generate("Once upon a time", max_new_tokens=50)
print(output)
```

### Customization

**Add new persona:**

```json
{
  "persona_datasets": {
    "pirate": {
      "description": "Talks like a pirate with 'arr' and nautical terms",
      "positive_texts": [
        "Arr matey! Ye be lookin' for treasure on the high seas!",
        "Shiver me timbers! That be a fine ship ye have there!",
        "Avast ye! Hand over the booty or walk the plank!"
      ],
      "negative_texts": [
        "Hello, how are you doing today?",
        "I think we should discuss this professionally.",
        "The weather is quite nice this afternoon."
      ]
    }
  }
}
```

**Try different layers:**

```python
for layer_idx in [8, 12, 16, 20, 24]:
    config['layer_idx'] = layer_idx
    config['output_dir'] = f"./results/layer{layer_idx}"
    experiment = PersonaTransferExperiment(config)
    experiment.run()
```

**Test multiple α values:**

```python
config['alpha_values'] = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
```

---

## Expected Results

### Success Criteria

**Minimum viable result:**

| Condition      | Coherence | Trait Strength | Collapse Rate |
|----------------|-----------|----------------|---------------|
| Baseline       | 4.5       | 1.2            | 0%            |
| Zero-pad       | 2.0       | 2.0            | 60%           |
| Interpolate    | 2.5       | 2.2            | 40%           |
| Random proj    | 1.3       | 1.0            | 90%           |
| **Ridge mapped** | **3.5+**  | **2.5+**       | **<30%**      |
| Oracle (native)| 4.4       | 4.0            | 5%            |

**Strong result:**

| Condition      | Coherence | Trait Strength | Collapse Rate |
|----------------|-----------|----------------|---------------|
| **Ridge mapped** | **4.0+**  | **3.5+**       | **<15%**      |

### Interpretation

1. **If mapped ≈ oracle:** Problem solved! Perfect transfer achieved.

2. **If mapped > naive >> random:** Partial success. Shows learned structure helps, even if not perfect.

3. **If mapped ≈ naive:** Method doesn't work. Possible causes:
   - Wrong layer (try layer ablation)
   - Insufficient training data (collect more samples)
   - Models too different (try more similar model pairs)
   - Extraction protocol mismatch

4. **If all methods collapse:** Problem with steering itself, not transfer:
   - Try lower α values
   - Check if persona vectors valid on source model
   - Verify steering hook working correctly

### Visualization Examples

**Figure 1: Main Results (Scatter Plot)**
- X-axis: Coherence (1-5)
- Y-axis: Trait Strength (1-5)
- Points: Each condition
- Shows: Mapped achieves high coherence AND trait strength

**Figure 2: Layer Ablation (Line Plot)**
- X-axis: Layer index (8, 12, 16, 20, 24)
- Y-axis: Coherence score
- Lines: Each method
- Shows: Middle layers (16-20) work best

**Figure 3: Training Data Ablation (Line Plot)**
- X-axis: Training samples (100, 500, 1k, 2k, 5k, 10k)
- Y-axis: Reconstruction error
- Shows: Elbow at ~2k samples

---

## Troubleshooting

### Common Issues

**Problem:** "CUDA out of memory"

**Solutions:**
- Reduce batch size in `extract_batch`
- Use FP16: `torch_dtype=torch.float16`
- Use gradient checkpointing
- Process in smaller chunks

---

**Problem:** "Ridge mapping has very high reconstruction error"

**Solutions:**
- Check if activations extracted correctly (same texts, same position)
- Try different λ values (maybe need stronger regularization)
- Verify models loaded correctly (check hidden dimensions)
- Compare to random baseline (should be much better than random)

---

**Problem:** "All methods including oracle cause collapse"

**Solutions:**
- α too high, try 0.1 - 0.5
- Verify persona vectors work on source model first
- Check steering hook is attached to correct layer module
- Test without steering (baseline) to verify model works

---

**Problem:** "Mapped vectors have low similarity to oracle"

**Solutions:**
- This is expected! Cosine similarity 0.3-0.6 can still work
- What matters is downstream performance, not similarity
- Check if mapped coherence > naive coherence
- Try different layers (16 might not be optimal)

---

**Problem:** "GPT-4 API calls failing"

**Solutions:**
- Check API key is set: `export OPENAI_API_KEY=sk-...`
- Verify billing is active
- Rate limit: add delays between calls
- Use `use_gpt4=False` for testing (skip expensive evaluation)

---

**Problem:** "Results are highly variable across seeds"

**Solutions:**
- This is normal for text generation
- Use more seeds (5-10 instead of 3)
- Report means ± std
- Consider increasing sample size

---

### Debugging Checklist

Before reporting issues, verify:

- [ ] Models loaded correctly (print hidden dimensions)
- [ ] Activations extracted at correct layer and position
- [ ] Persona vectors normalized to unit norm
- [ ] Ridge λ selected via cross-validation (not arbitrary)
- [ ] Steering hook attached to correct module
- [ ] Baseline (no steering) produces fluent text
- [ ] Oracle (native vector) works on target model
- [ ] Extraction protocol consistent everywhere

---

## Citation

If you use this code or methodology, please cite:

```bibtex
@article{yourname2024persona,
  title={Linear Mapping for Cross-Model Persona Transfer},
  author={Your Name},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2024}
}
```

---

## Acknowledgments

This methodology builds on:
- Steering vectors: [Subramani et al., 2022; Turner et al., 2023]
- Activation engineering: [Zou et al., 2023]
- Cross-lingual alignment: [Conneau et al., 2020]
- Ridge regression: [Hoerl & Kennard, 1970]

---

## Contact

For questions or issues:
- GitHub Issues: [link]
- Email: [your email]
- Twitter: [@handle]

---

*Last updated: 2024*
