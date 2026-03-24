# Cross-Model Persona Transfer

## Quick Start

### Run evaluations

**Same-model transfer:**
```bash
python src/evaluate_transfer.py --source llama-3.1-8b-instruct --target llama-3.1-8b-instruct --trait silly
```

**Cross-model transfer (same dimensions):**
```bash
python src/evaluate_transfer.py --source llama-3.1-8b-instruct --target mistral-7b-instruct-v0.3 --trait silly
```

**Cross-model transfer (different dimensions, automatic mapping):**
```bash
python src/evaluate_transfer.py --source qwen2.5-7b-instruct --target llama-3.1-8b-instruct --trait silly
```

## Models Supported
- `qwen2.5-7b-instruct` (3584 dims)
- `llama-3.1-8b-instruct` (4096 dims)
- `mistral-7b-instruct-v0.3` (4096 dims)
