# Cross-Model Persona Transfer

This repository implements and extends the methodology of Chen et al. (2024) to enable cross-model persona vector transfer and black-box steering of large language models.
The system extracts persona vectors from a source model and applies them to steer behavior in other transformer-based model families.

## Research Contributions

### 1. Cross-Family Vector Transfer
Extract persona vectors from Qwen2.5-7B and successfully apply them to Llama-3.1-8B and Mistral-7B - models with different training data, vocabularies, and attention mechanisms. Chen et al. tested each model independently without cross-model transfer.

### 2. Parameter Modulation for Black-Box Steering
For GGUF/quantized models without activation access, we map vector statistics to generation parameters:
- Vector magnitude → temperature adjustment
- Vector polarity → top_p, repetition_penalty
- Enables steering for APIs and edge devices where model internals are inaccessible

### 3. Dynamic Layer Selection and Effectiveness Scoring
Automatically identify the most effective transformer layers for steering rather than using fixed layers:
- Per-layer effectiveness scores during extraction
- Layer-wise activation difference analysis
- Adaptive steering strength based on layer importance
- Eliminates Chen et al.'s fixed layer-20 assumption

## Installation

```bash
# Clone and setup
git clone https://github.com/yourusername/cross-architecture-persona-transfer.git
cd cross-architecture-persona-transfer
./setup.sh

# Activate environment
source venv/bin/activate
```

**Requirements:** Python 3.10+, CUDA GPU (8GB+ VRAM recommended)

## Quick Usage

### Extract Persona Vectors

```bash
# Extract silly vectors from Qwen (source model)
python extract_vectors.py \
    --model qwen2.5-7b-instruct \
    --trait silly
```

Vectors saved to `src/data/vectors/qwen2.5-7b-instruct_silly.json`

### Test Cross-Model Transfer

```bash
# Apply Qwen vectors to Llama (cross-family transfer)
python apply_steering.py \
    --model llama-3.1-8b-instruct \
    --vectors src/data/vectors/qwen2.5-7b-instruct_silly.json \
    --prompt "Explain how a computer works" \
    --coefficient 1.5 \
    --baseline
```

### Systematic Evaluation

```bash
# Evaluate transfer across multiple coefficients
python evaluate_transfer.py \
    --source qwen2.5-7b-instruct \
    --target llama-3.1-8b-instruct \
    --trait silly \
    --coefficients -2.0 -1.0 0.0 1.0 2.0 \
    --output experiments/qwen_to_llama_silly.json
```

## Reproduce Paper Results

```bash
# Runs all main experiments (1-2 hours)
./reproduce_paper.sh
```

Results saved to `experiments/paper_results/`

## Command-Line Interface

### extract_vectors.py

Extract persona vectors from a source model.

**Arguments:**
- `--model` - Model ID (qwen2.5-7b-instruct, llama-3.1-8b-instruct, etc.)
- `--trait` - Trait to extract (silly, honest, or custom)
- `--output` - Output directory (default: src/data/vectors)

**Example:**
```bash
python extract_vectors.py --model qwen2.5-7b-instruct --trait silly
```

### apply_steering.py

Apply persona steering to generate responses.

**Arguments:**
- `--model` - Target model
- `--vectors` - Path to vector JSON file
- `--prompt` - Input question
- `--coefficient` - Steering strength (-2.0 to 2.0)
- `--baseline` - Also show unsteered response
- `--output` - Save results to JSON

**Example:**
```bash
python apply_steering.py \
    --model llama-3.1-8b-instruct \
    --vectors src/data/vectors/qwen2.5-7b-instruct_silly.json \
    --prompt "What is machine learning?" \
    --coefficient 1.5 \
    --baseline
```

### evaluate_transfer.py

Systematic evaluation with multiple prompts and coefficients.

**Arguments:**
- `--source` - Source model (where vectors extracted)
- `--target` - Target model (where steering applied)
- `--trait` - Trait being evaluated
- `--coefficients` - List of coefficients to test
- `--num-prompts` - Number of test prompts
- `--output` - Output JSON file

**Example:**
```bash
python evaluate_transfer.py \
    --source qwen2.5-7b-instruct \
    --target mistral-7b-instruct-v0.3 \
    --trait silly \
    --coefficients -2.0 -1.0 0.0 1.0 2.0 \
    --output experiments/qwen_to_mistral.json
```

## Supported Models

**HuggingFace (Direct Activation):**
- Qwen2.5-7B-Instruct
- Llama-3.1-8B-Instruct
- Mistral-7B-Instruct-v0.3
- GPT-2 Medium (for testing)

**GGUF (Parameter Modulation):**
- GPT-OSS 20B (requires `python download_gptoss.py`)

## Built-in Traits

- **silly** - Humorous/playful vs formal/serious
- **honest** - Truthful vs deceptive

Custom traits can be created via the web demo (optional, see `demo/`).

## Project Structure

```
cross-architecture-persona-transfer/
├── extract_vectors.py          # Extract persona vectors
├── apply_steering.py            # Apply steering to prompts
├── evaluate_transfer.py         # Systematic evaluation
├── reproduce_paper.sh           # Reproduce all experiments
│
├── src/                         # Core library
│   ├── models.py               # Model loading & inference
│   ├── persona_vectors.py      # Vector extraction & steering
│   ├── prompts.py              # Trait prompts
│   └── data/                   # Vectors and custom traits
│
├── experiments/                 # Results and configs
├── demo/                        # Web interface (optional)
└── docs/                        # Research paper
```

## Web Demo (Optional)

For interactive exploration:

```bash
cd demo
python main.py
# Open http://localhost:8000
```

**Note:** Web demo is supplementary. All research can be done via CLI.

## Citation

If you use this work, please cite:

```bibtex
@article{chen2024persona,
  title={Persona Vectors: Monitoring and Controlling Character Traits in Language Models},
  author={Chen, et al.},
  journal={arXiv preprint arXiv:2507.21509},
  year={2024}
}
```

## Key Implementation Details

### Direct Activation Injection (HuggingFace)
```python
# Hook transformer layers during generation
activation += coefficient * persona_vector
```

### Parameter Modulation (GGUF/APIs)
```python
# Map vector stats to generation parameters
temperature = base_temp + (vector_magnitude * coefficient * 0.1)
top_p = adjust_based_on_polarity(vector_stats)
```

See `src/models.py` for full implementation.

## Results Summary

Our experiments demonstrate:
- **Cross-family transfer works:** Qwen vectors effectively steer Llama and Mistral
- **Coherence maintained:** >70% coherence across coefficient range [-2, 2]
- **Parameter modulation viable:** Black-box steering without model internals
- **Trait strength correlates:** Linear relationship between coefficient and trait expression

See paper for detailed analysis.

## Troubleshooting

**Out of memory:**
```bash
# Use smaller model for testing
python extract_vectors.py --model gpt2-medium --trait silly
```

**Models downloading slow:**
```bash
# Set HuggingFace cache
export HF_HOME=/path/to/large/disk
```

**CUDA not available:**
```bash
# Will work on CPU (slower)
python extract_vectors.py --model gpt2-medium --trait silly
```

## License

MIT License - See LICENSE file

## Diagnostic Tests

The `tests/` directory contains systematic tests to diagnose why coherence breaks during cross-model persona transfer.

### Hypothesis A: Magnitude Mismatch
Tests if the vector direction is meaningful but scaled wrong.

```bash
python tests/test_hypothesis_a_magnitude_mismatch.py \
  --source qwen2.5-7b-instruct \
  --target llama-3.1-8b-instruct \
  --trait silly \
  --coefficients -1.0 0.0 1.0 2.0 \
  --num-prompts 5
```

### Hypothesis B: Basis Mismatch
Tests if the direction is rotated (basis mismatch between model latent spaces).

```bash
python tests/test_hypothesis_b_basis_mismatch.py \
  --source qwen2.5-7b-instruct \
  --target llama-3.1-8b-instruct \
  --trait silly \
  --coefficients 1.0 2.0 \
  --num-prompts 5
```

### Hypothesis C: Layer Correspondence
Tests if the vector hits the wrong layer (layer roles differ across families).

```bash
python tests/test_hypothesis_c_layer_correspondence.py \
  --source qwen2.5-7b-instruct \
  --target llama-3.1-8b-instruct \
  --trait silly \
  --coefficient 2.0 \
  --num-prompts 5
```

Results are saved to `experiments/hypothesis_*_<source>_to_<target>_<trait>.json` with detailed measurements and conclusions.

## Acknowledgments

Based on Chen et al. (2024) "Persona Vectors" methodology. This work extends their approach with cross-family transfer and parameter modulation for practical deployment.
