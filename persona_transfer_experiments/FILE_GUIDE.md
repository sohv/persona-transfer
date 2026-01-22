# Essential Files Guide

## Quick Reference: What Each File Does

### Core Implementation Files (in order of execution)

#### 1. `activation_extractor.py` (8.8 KB)
**What it does:** Extracts hidden state activations from language models consistently.

**Key class:** `ActivationExtractor`
```python
extractor = ActivationExtractor(model, tokenizer, layer_idx=16, device="cuda")
activation = extractor.extract(text, position='last')  # (hidden_dim,)
persona_vector = extractor.extract_persona_vector(positive_texts, negative_texts)
```

**Critical design decision:** Uses **last token position** because that's where steering is applied during generation.

**Theory:** Extraction protocol must match application context. During generation, steering happens at the last token before decoding, so we train on last-token activations.

**Related theory:** Contrastive learning for persona vectors:
```
v_persona = mean(activations_positive) - mean(activations_negative)
```

**Depends on:** PyTorch, Transformers
**Used by:** `ridge_mapping.py`, `run_experiment.py`

---

#### 2. `ridge_mapping.py` (12.9 KB)
**What it does:** Learns linear transformation W: R^d_source → R^d_target via ridge regression.

**Key class:** `RidgeMapping`
```python
mapping = RidgeMapping(d_source=3548, d_target=4096)
best_lambda, cv_errors = mapping.cross_validate(H_source, H_target)
v_target = mapping.transform(v_source, normalize=True)
```

**Critical design decision:** Uses **ridge regularization** to handle 14M parameters with only 5k samples.

**Theory:** Ridge regression solution:
```
W = (H_source^T H_source + λI)^{-1} H_source^T H_target

minimize ||H_target - H_source W||² + λ||W||²
```

With λ > 0, effective parameters ≈ effective rank (~1000-2000) << total parameters (14M).

**Why 5k samples is enough:**
- Without regularization: Need ~14M samples (one per parameter)
- With λ=1.0: Need ~5k samples (effective DOF ~1000)
- Cross-validation finds optimal λ

**Key method:** `analyze_mapping()` - Computes SVD to show effective rank:
```python
U, S, Vt = torch.svd(W)
effective_rank = (S > 0.01 * S[0]).sum()  # Typically ~1000-2000
```

**Depends on:** PyTorch, NumPy, scikit-learn
**Used by:** `run_experiment.py`

---

#### 3. `baseline_methods.py` (10.4 KB)
**What it does:** Implements naive transfer methods for fair comparison.

**Key class:** `BaselineTransfer`
```python
baseline = BaselineTransfer(d_source=3548, d_target=4096)
transferred = baseline.apply_all_baselines(
    v_source,
    v_oracle,
    ridge_mapping,
    normalize=True
)
```

**Methods implemented:**
1. **Zero-padding:** `[v_source, 0, 0, ..., 0]` - Simplest approach
2. **Interpolation:** Linear resampling to target dimension
3. **Random projection:** Random orthogonal matrix (control baseline, should fail)
4. **Ridge mapping:** Our learned method
5. **Oracle:** Native extraction from target model (upper bound)

**Critical design decision:** Must test **multiple baselines** to prove learned mapping isn't just lucky.

**Theory:** Random projection should NOT work (proves learned mapping captures structure). Oracle shows ceiling performance.

**Comparison metric:** Cosine similarity to oracle
```python
similarities = baseline.compare_to_oracle(v_source, v_oracle, ridge_mapping)
# Output: {'zero_pad': 0.23, 'random_proj': 0.01, 'ridge_mapped': 0.61, ...}
```

**Depends on:** PyTorch, NumPy
**Used by:** `run_experiment.py`

---

#### 4. `steered_generation.py` (11.9 KB)
**What it does:** Generates text with persona steering via activation patching.

**Key class:** `SteeredGenerator`
```python
generator = SteeredGenerator(model, tokenizer, device="cuda")
generator.register_steering(
    layer_idx=16,
    steering_vector=v_persona,
    alpha=2.0
)
output = generator.generate(prompt, max_new_tokens=100)
```

**Critical design decision:** Uses **PyTorch forward hooks** to intercept and modify hidden states during generation.

**Theory:** Activation steering equation:
```
h'_layer = h_layer + α * v_persona
```

Applied at every token generation step:
```python
for t in range(max_tokens):
    h_t = model(context)[:, -1, :]  # Last token
    h_t_steered = h_t + α * v_persona  # Add steering
    logits = lm_head(h_t_steered)
    next_token = sample(logits)
```

**Implementation via hook:**
```python
def steering_hook(module, input, output):
    hidden_states = output[0]  # (batch, seq, hidden_dim)
    steered = hidden_states + self.alpha * self.steering_vector
    return (steered,) + output[1:]

layer_module.register_forward_hook(steering_hook)
```

**Key feature:** Can dynamically change α without re-registering:
```python
generator.set_alpha(5.0)  # Stronger steering
```

**Depends on:** PyTorch, Transformers
**Used by:** `run_experiment.py`

---

#### 5. `evaluation.py` (17.2 KB)
**What it does:** Evaluates quality of steered generations.

**Key class:** `Evaluator`
```python
evaluator = Evaluator(model, tokenizer, device="cuda", use_gpt4=False)
result = evaluator.evaluate_single(
    prompt=prompt,
    output=output,
    condition="ridge_mapped",
    persona="silly"
)
```

**Metrics implemented:**

1. **Perplexity:** Model's confidence in its own output
   ```python
   perplexity = exp(negative_log_likelihood)
   ```
   Lower = more fluent. Uses target model as judge.

2. **Repetition collapse detection:**
   ```python
   unique_ngram_ratio = len(set(ngrams)) / len(ngrams)
   is_collapsed = unique_ngram_ratio < 0.5
   ```
   Common failure mode for naive transfer.

3. **GPT-4 coherence judge:** (optional, expensive)
   - Rates 1-5: 1 = gibberish, 5 = perfect
   - Gold standard from steering literature

4. **GPT-4 trait judge:** (optional, expensive)
   - Rates 1-5: 1 = trait absent, 5 = trait very strong
   - Validates steering worked

**Critical design decision:** Need both **coherence** (is it fluent?) AND **trait strength** (did steering work?).

**Theory:** Naive transfer often produces:
- Low coherence (collapsed, repetitive)
- High trait (steering too strong, destroys coherence)

Good transfer should have:
- High coherence (fluent, natural)
- Moderate-to-high trait (steering worked)

**Aggregation:**
```python
aggregated = evaluator.aggregate_results(eval_results)
# Output: {condition: {metric: mean_value, ...}, ...}
```

**Depends on:** PyTorch, Transformers, OpenAI API (optional)
**Used by:** `run_experiment.py`

---

#### 6. `run_experiment.py` (16.0 KB)
**What it does:** Orchestrates the complete experimental pipeline.

**Key class:** `PersonaTransferExperiment`
```python
experiment = PersonaTransferExperiment(config)
experiment.run()  # Executes full pipeline
```

**Pipeline stages:**

1. **Load models** (10 min)
   ```python
   self.load_models()
   # Loads Qwen and LLaMA, creates extractors
   ```

2. **Collect training data** (1 hour)
   ```python
   H_source, H_target = self.collect_training_data()
   # 5k texts from WikiText → paired activations
   ```

3. **Learn mapping** (5 min)
   ```python
   self.learn_mapping(H_source, H_target)
   # Cross-validates λ, fits W, analyzes via SVD
   ```

4. **Extract personas** (30 min)
   ```python
   persona_vectors = self.extract_persona_vectors()
   # From both models, applies all baselines
   ```

5. **Generate** (4 hours)
   ```python
   generation_results = self.run_generation_experiments(persona_vectors)
   # 3 personas × 5 conditions × 30 prompts × 5 α × 3 seeds = 6750 generations
   ```

6. **Evaluate** (2 hours)
   ```python
   eval_results, aggregated = self.evaluate_results(generation_results)
   # Computes all metrics, aggregates by condition
   ```

**Critical design decision:** **Sequential execution** - each stage depends on previous stage's output.

**Theory:** This is a **supervised learning** approach:
- Training: Learn W from paired activations
- Application: Map and evaluate persona vectors
- Validation: Compare to oracle (native target vectors)

**Outputs saved:**
- `training_data.pt`: Paired activations
- `ridge_mapping.pt`: Learned W
- `cv_results.json`: Cross-validation errors
- `persona_vectors.pt`: All extracted vectors
- `generation_results.json`: All generations
- `eval_results.json`: Per-generation metrics
- `aggregated_results.json`: Mean metrics by condition

**Depends on:** All above modules
**Used by:** Command line via `python run_experiment.py --config config.json`

---

### Configuration File

#### `example_config.json` (5.1 KB)
**What it does:** Specifies all experiment parameters.

**Key sections:**

1. **Models:**
   ```json
   {
     "source_model": "Qwen/Qwen-7B",
     "target_model": "meta-llama/Llama-2-7b-hf",
     "layer_idx": 16
   }
   ```

2. **Data:**
   ```json
   {
     "n_training_samples": 5000,
     "n_test_prompts": 30,
     "n_seeds": 3
   }
   ```

3. **Hyperparameters:**
   ```json
   {
     "lambda_values": [0.01, 0.1, 1.0, 10.0, 100.0],
     "alpha_values": [0.5, 1.0, 2.0, 3.0, 5.0]
   }
   ```

4. **Persona datasets:**
   ```json
   {
     "persona_datasets": {
       "silly": {
         "description": "Playful, humorous tone",
         "positive_texts": ["Happy text!", ...],
         "negative_texts": ["Serious text.", ...]
       }
     }
   }
   ```

**Critical parameters:**
- `layer_idx`: Which layer to extract/steer (16 is middle layer for most 7B models)
- `lambda_values`: Range for cross-validation (1.0 typically optimal)
- `alpha_values`: Steering strengths to test (2.0-3.0 typically best)

---

### Documentation Files

#### `METHODOLOGY.md` (25.0 KB)
**Comprehensive technical documentation covering:**
- Theoretical foundation (why ridge regression?)
- Mathematical derivations (closed-form solution)
- Implementation details (why last token? why same layer?)
- Experimental protocol (step-by-step guide)
- Expected results (success criteria)
- Troubleshooting (common issues and fixes)

**Key sections:**
- **Mathematical Details:** Full ridge regression derivation, effective DOF proof
- **Core Components:** Deep dive into each module's design
- **Experimental Pipeline:** Hour-by-hour timeline
- **Troubleshooting:** Debug checklist and common issues

#### `README.md` (8.2 KB)
**Quick start guide covering:**
- Installation and setup
- Quick start commands
- Core concepts summary
- File structure overview
- Expected timeline and hardware requirements
- Results interpretation

**Use this when:** You want to run experiments quickly without diving into theory.

#### `FILE_GUIDE.md` (This file)
**File-by-file reference guide covering:**
- What each file does
- Key classes and methods
- Design decisions and rationale
- Dependencies and relationships
- Theoretical foundations

**Use this when:** You want to understand a specific file or modify implementation.

---

## Dependency Graph

```
example_config.json
        ↓
run_experiment.py
        ↓
    ┌───┴───┬──────────┬─────────────┐
    ↓       ↓          ↓             ↓
activation_extractor.py  ridge_mapping.py  baseline_methods.py  steered_generation.py
    └───┬───┘          └──────┬──────┘     └─────────┬─────────┘
        └──────────────────────┴────────────────────────┐
                                                         ↓
                                                  evaluation.py
```

---

## Execution Flow

```
1. Load config
   ↓
2. Load models (Qwen, LLaMA)
   ↓
3. Create extractors (ActivationExtractor)
   ↓
4. Collect paired activations (WikiText-103)
   H_qwen: (5000, 3548)
   H_llama: (5000, 4096)
   ↓
5. Learn mapping (RidgeMapping)
   Cross-validate λ → Fit W: (3548, 4096) → Analyze SVD
   ↓
6. Extract personas (ActivationExtractor)
   v_qwen_silly, v_llama_silly_native, ...
   ↓
7. Apply baselines (BaselineTransfer)
   v_zero_pad, v_interpolate, v_random, v_ridge_mapped
   ↓
8. Generate text (SteeredGenerator)
   For each: persona × condition × prompt × α × seed
   ↓
9. Evaluate (Evaluator)
   Compute: perplexity, collapse, GPT-4 scores
   ↓
10. Aggregate results
    Group by condition → Compute means → Save JSON
```

---

## Key Design Decisions Summary

| Decision | File | Rationale |
|----------|------|-----------|
| Last token extraction | `activation_extractor.py` | Matches steering application during generation |
| Same layer everywhere | All files | Can't mix layer 16 map with layer 20 vectors |
| Ridge regularization | `ridge_mapping.py` | Handles 14M params with 5k samples via λ > 0 |
| Multiple baselines | `baseline_methods.py` | Prove learned mapping better than arbitrary |
| Forward hooks | `steered_generation.py` | Non-invasive steering without model retraining |
| Both coherence + trait | `evaluation.py` | Need fluency AND trait strength |
| Sequential pipeline | `run_experiment.py` | Each stage depends on previous output |

---

## Theory-to-Implementation Mapping

| Theoretical Concept | Implementation | File |
|---------------------|----------------|------|
| Contrastive persona extraction | `extract_persona_vector()` | `activation_extractor.py` |
| Ridge regression W = (X^T X + λI)^{-1} X^T Y | `fit()` method | `ridge_mapping.py` |
| Effective DOF = Σ σ²/(σ² + λ) | `analyze_mapping()` | `ridge_mapping.py` |
| Activation steering h' = h + αv | `SteeringHook.__call__()` | `steered_generation.py` |
| Perplexity = exp(NLL) | `compute_perplexity()` | `evaluation.py` |
| Repetition detection | `detect_repetition()` | `evaluation.py` |

---

## Quick Troubleshooting Guide

| Error | Likely File | Fix |
|-------|-------------|-----|
| "CUDA OOM" | `activation_extractor.py` | Reduce batch size in `extract_batch()` |
| High reconstruction error | `ridge_mapping.py` | Check λ, verify activations aligned |
| All methods collapse | `steered_generation.py` | Lower α values (try 0.1-0.5) |
| Low oracle similarity | `baseline_methods.py` | Expected! Check downstream performance |
| GPT-4 API error | `evaluation.py` | Check API key, set `use_gpt4=False` |

---

## Testing Individual Components

```bash
# Test extraction
python activation_extractor.py

# Test mapping
python ridge_mapping.py

# Test steering
python steered_generation.py

# Test evaluation
python evaluation.py

# Run full experiment
python run_experiment.py --config example_config.json
```

---

## Modification Guide

**Want to add a new persona?**
→ Edit `example_config.json`, add to `persona_datasets`

**Want to try different layers?**
→ Change `layer_idx` in config, run full pipeline

**Want to use different models?**
→ Change `source_model` and `target_model` in config

**Want to add a new baseline method?**
→ Add method to `BaselineTransfer` class in `baseline_methods.py`

**Want to change steering mechanism?**
→ Modify `SteeringHook.__call__()` in `steered_generation.py`

**Want to add new evaluation metrics?**
→ Add methods to `Evaluator` class in `evaluation.py`

---

## File Sizes and Complexity

| File | Lines | Size | Complexity |
|------|-------|------|------------|
| `activation_extractor.py` | ~250 | 8.8 KB | Medium |
| `ridge_mapping.py` | ~350 | 12.9 KB | High (math-heavy) |
| `baseline_methods.py` | ~300 | 10.4 KB | Low |
| `steered_generation.py` | ~350 | 11.9 KB | Medium (hooks) |
| `evaluation.py` | ~450 | 17.2 KB | Medium |
| `run_experiment.py` | ~400 | 16.0 KB | High (orchestration) |

**Total implementation:** ~2100 lines, ~77 KB

**Documentation:** ~1500 lines, ~38 KB

---

## For More Information

- **Quick start:** See `README.md`
- **Theory and derivations:** See `METHODOLOGY.md`
- **Specific implementation:** See this guide (`FILE_GUIDE.md`)
- **Configuration options:** See `example_config.json`

---

*Last updated: 2024-01-22*
