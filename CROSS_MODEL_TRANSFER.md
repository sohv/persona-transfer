# Cross-Model Persona Transfer

## Quick Start

### 1. Generate dimension mappings (one-time setup)
```bash
python create_mappings.py --all
```

### 2. Run evaluations

**Same-model transfer:**
```bash
python evaluate_transfer.py --source llama-3.1-8b-instruct --target llama-3.1-8b-instruct --trait silly
```

**Cross-model transfer (same dimensions):**
```bash
python evaluate_transfer.py --source llama-3.1-8b-instruct --target mistral-7b-instruct-v0.3 --trait silly
```

**Cross-model transfer (different dimensions, requires mapping):**
```bash
python evaluate_transfer.py --source qwen2.5-7b-instruct --target llama-3.1-8b-instruct --trait silly
```

## Models Supported
- `qwen2.5-7b-instruct` (3584 dims)
- `llama-3.1-8b-instruct` (4096 dims)
- `mistral-7b-instruct-v0.3` (4096 dims)
