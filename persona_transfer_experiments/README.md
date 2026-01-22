# Persona Transfer Between Language Models

Transfer persona vectors between different language models (e.g., Qwen → LLaMA) using learned linear mappings.

## Quick Start

```bash
# Install dependencies
pip install torch transformers datasets scikit-learn matplotlib

# Run sanity checks
python activation_extractor.py
python ridge_mapping.py
python steered_generation.py

# Run full experiment
python run_experiment.py --config example_config.json
```

## Core Idea

Extract persona vectors from Model A, learn a linear transformation to Model B's activation space via ridge regression, then apply to steer Model B's behavior.

**Key Innovation:** Ridge regression with cross-validation learns effective mapping from only 5k paired activations.

## Project Structure

```
persona_transfer_experiments/
├── activation_extractor.py      # Extract activations consistently
├── ridge_mapping.py             # Learn linear mapping via ridge regression
├── baseline_methods.py          # Naive transfer baselines (zero-pad, interpolate, etc.)
├── steered_generation.py        # Generate text with steering
├── evaluation.py                # Evaluate coherence and trait strength
├── run_experiment.py            # Main orchestrator
├── example_config.json          # Configuration template
├── METHODOLOGY.md               # Complete documentation (theory + implementation)
└── README.md                    # This file
```

## Essential Files

### 1. `activation_extractor.py`
**Purpose:** Consistent extraction protocol
**Key:** Uses last token position (matches steering during generation)
**Theory:** Extraction method must match application method

### 2. `ridge_mapping.py`
**Purpose:** Learn W: R^3548 → R^4096 via ridge regression
**Key:** Cross-validates λ, handles 14M parameters with 5k samples via regularization
**Theory:** Effective parameters << total parameters due to ridge penalty

### 3. `baseline_methods.py`
**Purpose:** Fair comparison baselines
**Key:** Tests zero-pad, interpolation, random projection, and learned mapping
**Theory:** Must prove learned mapping better than arbitrary transforms

### 4. `steered_generation.py`
**Purpose:** Generate with activation steering
**Key:** Adds persona vector to hidden states via forward hooks
**Theory:** h' = h + α·v modifies behavior without retraining

### 5. `evaluation.py`
**Purpose:** Measure coherence and trait strength
**Key:** Perplexity, collapse detection, optional GPT-4 judge
**Theory:** Need both coherence (fluency) AND trait (steering worked)

### 6. `run_experiment.py`
**Purpose:** Orchestrate full pipeline
**Key:** Handles data collection → mapping → generation → evaluation
**Theory:** Each step depends on previous (order matters)

## Theory Summary

### The Problem
Persona vectors are model-specific and can't be directly transferred due to:
1. Dimensional mismatch (Qwen: 3548-d, LLaMA: 4096-d)
2. Different representational bases (even at same dimension)

### The Solution
Learn linear mapping W that aligns activation spaces:

```
Given: H_source (N×3548), H_target (N×4096) from same texts
Solve: W = argmin ||H_target - H_source W||² + λ||W||²
Apply: v_target = v_source @ W
```

### Why This Works
- Ridge regularization (λ > 0) reduces effective parameters to ~1000-2000
- Only need 5k samples, not 14M
- Linear maps preserve semantic structure if models aligned
- Closed-form solution (no training loop)

### Success Criteria
**Mapped** method should:
- Coherence > 3.5 (vs naive ~2.0)
- Trait strength > 2.5 (vs baseline ~1.2)
- Collapse rate < 30% (vs naive ~80%)

## Key Design Decisions

### 1. Last Token Extraction (NOT Mean Pooling)
**Rationale:** Steering adds vector to last token during generation
```python
# Generation loop:
h_last = model(context)[:, -1, :]  # Last token
h_steered = h_last + α * v_persona  # Apply here
```
Training on last token matches application distribution.

### 2. Same Layer Everywhere
**Rationale:** Can't mix layer 16 mapping with layer 20 vectors
```python
# Correct:
mapping = learn_mapping(layer=16)
v_qwen = extract_persona(qwen, layer=16)
v_llama = v_qwen @ mapping

# Wrong:
mapping = learn_mapping(layer=16)
v_qwen = extract_persona(qwen, layer=20)  # ❌ Different layer!
```

### 3. Multiple Baselines Required
**Rationale:** Must prove learned mapping better than arbitrary transforms
- Zero-pad: Simplest, might partially work
- Interpolation: Smooth but arbitrary
- Random projection: Should fail (control)
- Ridge: Should win (our method)
- Oracle: Upper bound (native target vector)

### 4. Ridge Regularization
**Rationale:** 14M parameters but only 5k samples
- Without λ: Overfit, poor generalization
- With λ: Effective parameters ~1000, well-conditioned
- Cross-validate λ: Find sweet spot

## Configuration Guide

**Minimal config:**
```json
{
  "source_model": "Qwen/Qwen-7B",
  "target_model": "meta-llama/Llama-2-7b-hf",
  "layer_idx": 16,
  "output_dir": "./results",
  "persona_datasets": {
    "silly": {
      "positive_texts": ["Happy playful text!", ...],
      "negative_texts": ["Serious formal text.", ...]
    }
  },
  "test_prompts": ["Tell me about AI.", ...]
}
```

**Full options:**
- `n_training_samples`: 5000 (default), more = better but slower
- `lambda_values`: [0.01, 0.1, 1.0, 10.0] for cross-validation
- `alpha_values`: [0.5, 1.0, 2.0, 5.0] steering strengths to test
- `n_test_prompts`: 30 prompts × 3 seeds = 90 generations per condition
- `use_gpt4`: false (default), true = expensive but gold standard evaluation

## Expected Timeline

| Phase | Duration | Output |
|-------|----------|--------|
| Load models | 10 min | Models in memory |
| Collect training data | 1 hour | 5k paired activations |
| Learn mapping | 5 min | W matrix, CV results |
| Extract personas | 30 min | Vectors for all methods |
| Generate | 4 hours | 6750 generations |
| Evaluate | 2 hours | Metrics, tables |
| **Total** | **~8 hours** | Complete results |

## Hardware Requirements

**Minimum:**
- GPU: 24GB VRAM (A5000, RTX 3090)
- RAM: 32GB
- Storage: 50GB

**Recommended:**
- GPU: 40GB VRAM (A100)
- RAM: 64GB
- Storage: 100GB

**Optimizations:**
- Use FP16: `torch_dtype=torch.float16`
- Batch generation where possible
- Offload inactive model to CPU

## Common Issues

**"CUDA out of memory"**
→ Reduce batch size, use FP16, offload to CPU

**"Ridge error very high"**
→ Check activation extraction, try different λ

**"All methods collapse"**
→ α too high, try 0.1-0.5

**"Low oracle similarity"**
→ Expected! Check downstream performance instead

## Results Interpretation

### Scenario 1: Clear Success ✓
```
Condition      | Coherence | Trait | Collapse
Ridge mapped   |    4.2    |  3.7  |   8%
Oracle native  |    4.5    |  4.1  |   3%
Naive methods  |    2.0    |  2.0  |  70%
```
**Conclusion:** Learned mapping successfully transfers persona while preserving coherence. Approaches oracle performance. **Publishable!**

### Scenario 2: Partial Success ⚠️
```
Condition      | Coherence | Trait | Collapse
Ridge mapped   |    3.6    |  2.8  |  25%
Oracle native  |    4.5    |  4.1  |   3%
Naive methods  |    2.0    |  2.0  |  70%
```
**Conclusion:** Improvement over naive, but gap to oracle. Investigate: layer choice, more data, model similarity. **Still publishable** with analysis of limitations.

### Scenario 3: Failure ✗
```
Condition      | Coherence | Trait | Collapse
Ridge mapped   |    2.1    |  2.0  |  65%
Oracle native  |    4.5    |  4.1  |   3%
Naive methods  |    2.0    |  2.0  |  70%
```
**Conclusion:** No improvement. Debug: wrong layer, insufficient data, models too different, extraction bug. Can publish as **negative result** with thorough analysis.

## For More Details

See [METHODOLOGY.md](METHODOLOGY.md) for:
- Complete theoretical derivation
- Detailed implementation notes
- Mathematical proofs
- Troubleshooting guide
- Advanced usage examples

## Citation

```bibtex
@article{yourname2024persona,
  title={Linear Mapping for Cross-Model Persona Transfer},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

## License

MIT License - See LICENSE file

## Contact

GitHub Issues: [your repo]
Email: [your email]
