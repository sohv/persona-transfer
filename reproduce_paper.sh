#!/bin/bash
# Reproduce key results from the paper

set -e

echo "Cross-Model Persona Transfer - Reproduce Paper Results"
echo "========================================================"
echo ""
echo "This script will reproduce the main experiments from the paper."
echo "Estimated time: 1-2 hours depending on GPU"
echo ""
read -p "Continue? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 0
fi

# Create output directory
mkdir -p experiments/paper_results
RESULTS_DIR="experiments/paper_results"

echo ""
echo "Step 1: Extract persona vectors from Qwen"
echo "==========================================="
echo ""

# Extract silly vectors from Qwen
python extract_vectors.py \
    --model qwen2.5-7b-instruct \
    --trait silly \
    --output src/data/vectors

echo ""
echo "Step 2: Evaluate Cross-Family Transfer"
echo "======================================"
echo ""

# Qwen -> Llama (Cross-family transfer)
echo "Testing: Qwen -> Llama-3.1-8B..."
python evaluate_transfer.py \
    --source qwen2.5-7b-instruct \
    --target llama-3.1-8b-instruct \
    --trait silly \
    --coefficients -2.0 -1.0 0.0 1.0 2.0 \
    --num-prompts 10 \
    --output $RESULTS_DIR/qwen_to_llama_silly.json

# Qwen -> Mistral (Cross-family transfer)
echo ""
echo "Testing: Qwen -> Mistral-7B..."
python evaluate_transfer.py \
    --source qwen2.5-7b-instruct \
    --target mistral-7b-instruct-v0.3 \
    --trait silly \
    --coefficients -2.0 -1.0 0.0 1.0 2.0 \
    --num-prompts 10 \
    --output $RESULTS_DIR/qwen_to_mistral_silly.json

echo ""
echo "Step 3: Test Parameter Modulation (GGUF)"
echo "========================================"
echo ""
echo "Note: Requires GPT-OSS 20B GGUF model"
echo "If not available, this will be skipped"

if [ -f "models/openai_gpt-oss-20b-Q4_K_M.gguf" ]; then
    echo "Testing: Qwen -> GPT-OSS 20B (parameter modulation)..."
    python evaluate_transfer.py \
        --source qwen2.5-7b-instruct \
        --target gpt-oss-20b \
        --trait silly \
        --coefficients -1.0 0.0 1.0 \
        --num-prompts 5 \
        --output $RESULTS_DIR/qwen_to_gptoss_silly.json
else
    echo "GPT-OSS 20B not found, skipping..."
    echo "Run 'python download_gptoss.py' to download"
fi

echo ""
echo "Step 4: Qualitative Examples"
echo "============================"
echo ""

echo "Example 1: Baseline (no steering)"
python apply_steering.py \
    --model llama-3.1-8b-instruct \
    --vectors src/data/vectors/qwen2.5-7b-instruct_silly.json \
    --prompt "Explain how photosynthesis works" \
    --coefficient 0.0

echo ""
echo "Example 2: Positive steering (silly)"
python apply_steering.py \
    --model llama-3.1-8b-instruct \
    --vectors src/data/vectors/qwen2.5-7b-instruct_silly.json \
    --prompt "Explain how photosynthesis works" \
    --coefficient 1.5

echo ""
echo "Example 3: Negative steering (serious)"
python apply_steering.py \
    --model llama-3.1-8b-instruct \
    --vectors src/data/vectors/qwen2.5-7b-instruct_silly.json \
    --prompt "Explain how photosynthesis works" \
    --coefficient -1.5

echo ""
echo "========================================================"
echo "Reproduction complete!"
echo ""
echo "Results saved to: $RESULTS_DIR/"
echo ""
echo "Files:"
ls -lh $RESULTS_DIR/
echo ""
echo "To analyze results, load JSON files and compute metrics."
echo "========================================================"
