#!/bin/bash
# Complete setup and run script for any GPU server
# Just upload this entire repo and run: bash setup_and_run.sh

set -e

echo "=========================================="
echo "VLM Training - Auto Setup and Run"
echo "=========================================="

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 not found!"
    exit 1
fi

# Create virtual environment if needed
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel -q

# Install PyTorch with CUDA (will auto-detect CUDA version)
echo "Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 -q || \
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -q || \
pip install torch torchvision torchaudio -q

# Install other dependencies
echo "Installing dependencies..."
pip install -r requirements.txt -q

# Make scripts executable
chmod +x run_gpu_training.py 2>/dev/null || true

# Check GPU
echo ""
python3 -c "import torch; print('GPU Available:', torch.cuda.is_available()); print('GPU Count:', torch.cuda.device_count() if torch.cuda.is_available() else 0)"

# Run training
echo ""
echo "Starting training..."
echo ""

python3 run_gpu_training.py

echo ""
echo "Done! Check ./checkpoints for saved models"

