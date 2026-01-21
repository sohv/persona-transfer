# Code Documentation - Persona Transfer Project

This document provides an overview of all important code files in the persona-transfer project. The project implements cross-model persona vector transfer to steer language model behavior while maintaining generalization across different model families.

## Project Overview

The persona-transfer system enables:
1. **Cross-Family Vector Transfer**: Extract persona vectors from one model (e.g., Qwen) and apply them to different models (e.g., Llama, Mistral)
2. **Black-Box Steering**: Map vector statistics to generation parameters for models without activation access
3. **Dynamic Layer Selection**: Automatically identify optimal transformer layers for steering

---

## Core Source Files

### 📄 [src/models.py](src/models.py)

**Purpose**: Model loading, inference, and persona vector application for different model types.

**Key Components**:
- **Model Registry**: Manages available models (Qwen, Llama, Mistral, GPT-2, DialoGPT)
  - Supports both HuggingFace transformers and GGUF format via llama.cpp
  - Configures quantization, device mapping, and max sequence length
  
- **Model Loading** (`_load_model`):
  - Handles 4-bit quantization using BitsAndBytesConfig for memory efficiency
  - Uses Accelerate for distributed inference
  - Loads tokenizers and sets up generation configurations

- **Inference Functions**:
  - `get_model_response()`: Generate responses with optional activation extraction
  - `_generate_text_response()`: Text generation with custom stopping criteria
  - Supports capturing intermediate layer activations for vector extraction

- **Parameter Modulation Steering** (Black-Box Method):
  - Maps persona vector statistics to generation parameters:
    - **Vector magnitude** → temperature adjustment (controls randomness)
    - **Vector polarity** → top_p and repetition_penalty (controls diversity)
  - Enables steering without requiring access to model activations (for APIs, edge devices)

- **Activation Handling**:
  - Custom hooks to capture transformer layer outputs during forward pass
  - Layer-specific activation extraction for vector computation

---

### 📄 [src/persona_vectors.py](src/persona_vectors.py)

**Purpose**: Core functions for generating, storing, and manipulating persona vectors.

**Key Components**:
- **Persona Vector Generation** (`generate_persona_vectors`):
  - Processes prompt pairs (positive/negative trait examples)
  - For each evaluation question:
    1. Generates responses for both positive and negative prompts
    2. Captures model activations at specified layers
    3. Computes vectors as activation differences: `vector = activation_positive - activation_negative`
  - Scores effectiveness of each layer for the trait
  - Handles circuit-breaker logic to prevent infinite loops on API failures

- **Vector Storage** (`NumpyEncoder`, `save_vectors`):
  - JSON serialization of numpy arrays for persistent storage
  - Stores vectors with metadata: layer indices, trait name, model info, timestamps
  - Saves generated responses for analysis

- **Vector Loading** (`load_persona_vectors`):
  - Loads pre-computed vectors from JSON files
  - Parses vector metadata for target layer selection

- **Effectiveness Scoring**:
  - Per-layer scores indicating which layers best capture trait-specific information
  - Used for adaptive steering strength based on layer importance

- **Vector Application**:
  - Applies loaded vectors to target model during inference
  - Supports steering coefficients (scaling factor for vector influence)

---

### 📄 [src/prompts.py](src/prompts.py)

**Purpose**: Trait definitions, prompt pairs, and evaluation questions for persona extraction.

**Key Components**:
- **Built-in Traits**:
  - **Silly** (humorous vs. serious): 5 prompt pairs testing different aspects of humor
  - **Dishonest** (deceptive vs. truthful): 5 prompt pairs testing honesty/accuracy
  - **Superficial** (surface-level vs. deep): Prompt pairs testing depth of analysis

- **Prompt Pairs Structure**:
  - Each trait has positive/negative prompt pairs
  - Positive prompt: instructs model to exhibit the trait (e.g., "be silly")
  - Negative prompt: instructs opposite behavior (e.g., "be serious")
  - Vectors computed as difference in activations between these conditions

- **Evaluation Questions**:
  - Generic questions asked under positive/negative conditions
  - Used to test trait manifestation (e.g., "Explain photosynthesis")
  - Results show how model behavior changes under trait steering

- **Custom Trait Support**:
  - `load_custom_traits()`: Reads custom traits from `data/custom_traits.json`
  - Allows users to define new traits with custom prompt pairs and questions
  - Supports Chen et al. evaluation methodology with custom eval prompts

---

## Main Execution Scripts

### 🐍 [extract_vectors.py](extract_vectors.py)

**Purpose**: Extract persona vectors from a source model.

**Workflow**:
1. Accepts model ID and trait as arguments
2. Loads prompt pairs and evaluation questions for the trait
3. Calls `generate_persona_vectors()` from `persona_vectors.py`
4. Saves extracted vectors to JSON file

**Usage**:
```bash
python extract_vectors.py --model qwen2.5-7b-instruct --trait silly
python extract_vectors.py --model llama-3.1-8b-instruct --trait honest --output vectors/
```

**Output**: JSON file containing vectors, layer effectiveness scores, and metadata

---

### 🐍 [apply_steering.py](apply_steering.py)

**Purpose**: Apply pre-extracted persona vectors to steer model behavior.

**Workflow**:
1. Loads target model and pre-extracted vectors
2. Accepts user prompt and steering coefficient
3. Applies vectors to model activations during inference
4. Generates steered response
5. Optionally generates baseline response for comparison

**Key Features**:
- **Cross-Model Transfer**: Apply vectors from one model to another (e.g., Qwen vectors on Llama)
- **Steering Coefficient**: Control vector influence (-2.0 to 2.0 range)
- **Negative Steering**: Negative coefficients exhibit opposite trait
- **Baseline Comparison**: Generate both steered and unsteered responses

**Usage**:
```bash
python apply_steering.py --model llama-3.1-8b-instruct \
    --vectors src/data/vectors/qwen2.5-7b-instruct_silly.json \
    --prompt "Explain quantum physics" \
    --coefficient 1.5

# With baseline comparison
python apply_steering.py --model mistral-7b-instruct-v0.3 \
    --vectors vectors/honest.json \
    --prompt "What do you think about AI?" \
    --coefficient 1.0 --baseline
```

---

### 🐍 [evaluate_transfer.py](evaluate_transfer.py)

**Purpose**: Systematically evaluate cross-model persona transfer effectiveness.

**Workflow**:
1. Takes source model, target model, trait, and coefficient range as input
2. Loads source model's extracted vectors
3. Generates responses across multiple prompts at different steering strengths
4. Computes metrics:
   - **Trait Manifestation**: How strongly is the trait exhibited in responses?
   - **Coherence Score**: Quality of generated text (penalizes gibberish, rewards complete sentences)
   - **Transfer Effectiveness**: Success rate of cross-model application

5. Saves results with detailed metrics and response samples

**Usage**:
```bash
python evaluate_transfer.py --source qwen2.5-7b-instruct \
    --target llama-3.1-8b-instruct \
    --trait silly \
    --coefficients -2.0 -1.0 0.0 1.0 2.0
```

**Output**: JSON file with per-prompt results, metrics, and statistical analysis

---

### 🐍 [download_gptoss.py](download_gptoss.py)

**Purpose**: Download GPT-OSS 20B model in GGUF quantized format.

**Key Features**:
- Downloads 12GB quantized model file from HuggingFace Hub
- Supports resumable downloads with progress bar
- Detects existing downloads to prevent re-downloading
- Stores model in `models/` directory for llama.cpp integration

**Usage**:
```bash
python download_gptoss.py
```

**Note**: GGUF format enables inference on CPU and edge devices with parameter modulation steering

---

## Automation & Setup

### 📜 [reproduce_paper.sh](reproduce_paper.sh)

**Purpose**: Automated script to reproduce main paper results.

**Workflow**:
1. Extracts persona vectors from Qwen2.5-7B
2. Tests cross-family transfer:
   - Qwen → Llama-3.1-8B
   - Qwen → Mistral-7B
3. Tests parameter modulation steering on GPT-OSS 20B
4. Generates comprehensive evaluation reports

**Execution Time**: 1-2 hours on GPU

**Output**: Results saved to `experiments/paper_results/`

### 📜 [setup.sh](setup.sh)

**Purpose**: Initialize development environment.

**Setup Steps**:
- Creates Python virtual environment
- Installs dependencies from `requirements.txt`
- Configures environment variables
- Downloads/prepares models if needed

**Usage**:
```bash
./setup.sh
source venv/bin/activate
```

---

## Configuration & Data Files

### 📋 [requirements.txt](requirements.txt)

**Key Dependencies**:
- `transformers`: HuggingFace model loading and inference
- `torch`: PyTorch for GPU acceleration
- `accelerate`: Distributed inference utilities
- `llama-cpp-python`: GGUF model support
- `bitsandbytes`: 4-bit quantization
- `python-dotenv`: Environment variable management
- `pyyaml`: Configuration file parsing

---

### 📁 Directory Structure

```
src/data/
├── custom_traits.json          # User-defined traits (prompt pairs + questions)
└── vectors/
    ├── qwen2.5-7b-instruct_silly.json
    ├── qwen2.5-7b-instruct_dishonest.json
    └── [other extracted vectors]

experiments/
├── cross_family_eval.yaml       # Configuration for cross-family evaluation
└── paper_results/               # Output from reproduce_paper.sh
    ├── qwen_to_llama_silly.json
    ├── qwen_to_mistral_silly.json
    └── [other evaluation results]

models/
└── openai_gpt-oss-20b-Q4_K_M.gguf  # Downloaded GGUF model
```

---

## Data Flow Diagram

### Vector Extraction
```
extract_vectors.py
    ↓
Load model (models.py) + trait prompts (prompts.py)
    ↓
For each prompt pair & evaluation question:
    ├─ Generate positive response (capture activations)
    ├─ Generate negative response (capture activations)
    └─ Compute vector = pos_activations - neg_activations
    ↓
Score layer effectiveness
    ↓
Save vectors to JSON (persona_vectors.py)
```

### Vector Application (Cross-Model)
```
apply_steering.py
    ↓
Load target model + extracted vectors (from different source model)
    ↓
User provides: prompt + steering coefficient
    ↓
During inference:
    ├─ Apply vector: activation += coefficient × vector
    ├─ Or use parameter modulation for black-box models
    └─ Generate steered response
    ↓
Output: [baseline response] + [steered response]
```

### Evaluation
```
evaluate_transfer.py
    ↓
Load source vectors + target model
    ↓
For each prompt and coefficient:
    ├─ Generate steered response
    ├─ Compute trait manifestation score
    ├─ Compute coherence score
    └─ Collect metrics
    ↓
Statistical analysis + visualization
    ↓
Save detailed report to JSON
```

---

## Key Innovation: Parameter Modulation for Black-Box Steering

For models without activation access (APIs, GGUF quantized models):

**Vector Statistics → Generation Parameters Mapping**:
| Vector Property | Parameter | Effect |
|---|---|---|
| Magnitude (norm) | Temperature | Controls output randomness |
| Polarity (pos/neg) | Top-p | Controls output diversity |
| Direction | Repetition Penalty | Prevents repetition |
| Coefficient × Stats | Combined scaling | Overall steering strength |

**Benefits**:
- Works with any model through text-generation API
- No need for model internals/activation access
- Enables edge device and API-based steering
- Computationally efficient

---

## Typical Workflows

### Extract and Apply Vectors (Same Model)
```bash
# Extract silly vectors from Qwen
python extract_vectors.py --model qwen2.5-7b-instruct --trait silly

# Apply to Qwen (within same model)
python apply_steering.py --model qwen2.5-7b-instruct \
    --vectors src/data/vectors/qwen2.5-7b-instruct_silly.json \
    --prompt "What's your favorite food?" --coefficient 1.5
```

### Cross-Model Transfer (Main Innovation)
```bash
# Extract from Qwen
python extract_vectors.py --model qwen2.5-7b-instruct --trait silly

# Apply to Llama (different model)
python apply_steering.py --model llama-3.1-8b-instruct \
    --vectors src/data/vectors/qwen2.5-7b-instruct_silly.json \
    --prompt "Explain gravity" --coefficient 1.5

# Apply to Mistral (another different model)
python apply_steering.py --model mistral-7b-instruct-v0.3 \
    --vectors src/data/vectors/qwen2.5-7b-instruct_silly.json \
    --prompt "Tell me about clouds" --coefficient 1.5
```

### Systematic Evaluation
```bash
# Evaluate Qwen→Llama transfer across multiple conditions
python evaluate_transfer.py \
    --source qwen2.5-7b-instruct \
    --target llama-3.1-8b-instruct \
    --trait silly \
    --coefficients -2.0 -1.0 0.0 1.0 2.0 \
    --num-prompts 20 \
    --output results/qwen_to_llama.json
```

---

## Summary Table

| File | Type | Purpose | Key Functions |
|---|---|---|---|
| `models.py` | Module | Model loading & inference | `_load_model`, `get_model_response`, parameter modulation |
| `persona_vectors.py` | Module | Vector generation & storage | `generate_persona_vectors`, `load_persona_vectors`, effectiveness scoring |
| `prompts.py` | Module | Trait definitions | `get_prompt_pairs`, `get_evaluation_questions`, custom trait loading |
| `extract_vectors.py` | Script | Vector extraction | CLI entry point for extraction |
| `apply_steering.py` | Script | Vector application | CLI entry point for steering |
| `evaluate_transfer.py` | Script | Systematic evaluation | Cross-model effectiveness testing |
| `download_gptoss.py` | Script | Model downloading | Download GGUF model for llama.cpp |
| `reproduce_paper.sh` | Bash | Automation | Reproduce all paper experiments |
| `setup.sh` | Bash | Environment setup | Virtual environment + dependencies |

