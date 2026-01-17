#!/bin/bash
set -e

echo "Cross-Model Persona Vector Transfer System Setup"
echo "=================================================="
echo ""

# Check Python version
echo "Checking Python installation..."
if command -v python3.12 &> /dev/null; then
    PYTHON_CMD=python3.12
    echo "Found Python 3.12"
elif command -v python3.11 &> /dev/null; then
    PYTHON_CMD=python3.11
    echo "Found Python 3.11"
elif command -v python3.10 &> /dev/null; then
    PYTHON_CMD=python3.10
    echo "Found Python 3.10"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
    echo "Found Python 3"
else
    echo "ERROR: Python 3.10+ required. Please install Python first."
    exit 1
fi

# Verify CUDA availability (optional but recommended)
echo ""
echo "Checking for CUDA..."
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "WARNING: CUDA not detected. System will use CPU (significantly slower)."
    echo "For GPU acceleration, install CUDA toolkit: https://developer.nvidia.com/cuda-downloads"
fi

# Create virtual environment
echo ""
echo "Creating virtual environment..."
if [ ! -d "venv" ]; then
    $PYTHON_CMD -m venv venv
    echo "Virtual environment created"
else
    echo "Virtual environment already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA support
echo ""
echo "Installing PyTorch with CUDA support..."
echo "If you don't have CUDA, PyTorch will work with CPU (slower)"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install llama-cpp-python with CUDA support
echo ""
echo "Installing llama-cpp-python with CUDA support..."
echo "This may take several minutes to compile..."
if command -v nvcc &> /dev/null; then
    # CUDA is available, build with CUBLAS
    CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir
    echo "Built with CUDA acceleration"
else
    # No CUDA, build CPU-only version
    pip install llama-cpp-python
    echo "Built CPU-only version (install CUDA toolkit for GPU acceleration)"
fi

# Install other requirements
echo ""
echo "Installing remaining dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo ""
echo "Creating data directories..."
mkdir -p src/data/vectors
mkdir -p src/data/responses
mkdir -p models
mkdir -p experiments
mkdir -p logs

echo ""
echo "=================================================="
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Download GPT-OSS 20B model (optional):"
echo "   python download_gptoss.py"
echo ""
echo "2. Start the application:"
echo "   source venv/bin/activate"
echo "   cd src"
echo "   python main.py"
echo ""
echo "3. Open your browser to http://localhost:8000"
echo ""
echo "Note: Models will be downloaded automatically on first use."
echo "      Qwen2.5-7B (~15GB), Llama-3.1-8B (~16GB), Mistral-7B (~15GB)"
echo "=================================================="
